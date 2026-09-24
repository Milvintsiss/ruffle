//! Special handling for AVM2 orphan objects

use crate::context::UpdateContext;
use crate::display_object::{DisplayObject, DisplayObjectWeak, TDisplayObject};
use crate::frame_lifecycle::{is_idle_tree, set_tree_asleep};
use fnv::FnvHashSet;
use gc_arena::{Collect, Mutation};
use std::cell::RefCell;
use std::rc::Rc;

thread_local! {
    /// Addresses of asleep orphans woken since the queue was last drained
    /// (`DisplayObjectBase::wake_orphan`, perf-lab/ruffle-patches/0005).
    static WOKEN: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

/// Queues an asleep orphan for the frame passes again (perf-lab/ruffle-patches/0005).
pub fn queue_orphan_wake(addr: usize) {
    WOKEN.with(|w| w.borrow_mut().push(addr));
}

/// One object of the orphan list.
#[derive(Clone, Copy, Collect)]
#[collect(no_drop)]
struct OrphanEntry<'gc> {
    obj: DisplayObjectWeak<'gc>,
    /// The object was an idle orphan at the last `cleanup_dead_orphans` and its `ORPHAN_ASLEEP`
    /// flag is set: the frame passes skip it without looking at it, unless it has been woken
    /// since (perf-lab/ruffle-patches/0005). The entry keeps its place in the list, so the
    /// order in which orphans are run is unchanged.
    #[collect(require_static)]
    asleep: bool,
}

/// The list of 'orphan' objects - these objects have no parent,
/// so we need to manually run their frames in `run_all_phases_avm2` to match
/// Flash's behavior. Clips are added to this list with `add_orphan_movie`.
/// and are removed automatically by `cleanup_dead_orphans`.
///
/// We store `DisplayObjectWeak`, since we don't want to keep these objects
/// alive if they would otherwise be garbage-collected. The movie will
/// stop ticking whenever garbage collection runs if there are no more
/// strong references around (this matches Flash's behavior).
#[derive(Collect)]
#[collect(no_drop)]
pub struct OrphanManager<'gc> {
    orphans: Rc<Vec<OrphanEntry<'gc>>>,
    /// Addresses of the objects currently in `orphans`, so that `add_orphan_obj`
    /// does not have to scan the whole list (it is called for every MovieClip
    /// removed from its parent by script). Rebuilt by `cleanup_dead_orphans`.
    #[collect(require_static)]
    listed: FnvHashSet<usize>,
    /// Addresses woken since the last `cleanup_dead_orphans` that ran outside the passes
    /// (perf-lab/ruffle-patches/0005).
    #[collect(require_static)]
    woken: FnvHashSet<usize>,
    /// Frames since the asleep entries were last all checked (dead objects leave the list then).
    #[collect(require_static)]
    since_full_check: u32,
    /// `each_orphan_obj` passes in progress. A pass goes through a snapshot of the list, where
    /// an entry woken in the meantime still says asleep: only `woken` makes the pass run it.
    #[collect(require_static)]
    passes: u32,
}

impl<'gc> OrphanManager<'gc> {
    fn orphans_mut(&mut self) -> &mut Vec<OrphanEntry<'gc>> {
        Rc::make_mut(&mut self.orphans)
    }

    /// Moves the addresses queued by `DisplayObjectBase::wake_orphan` into `woken`.
    fn drain_woken(&mut self) {
        WOKEN.with(|w| {
            let mut w = w.borrow_mut();
            if !w.is_empty() {
                self.woken.extend(w.drain(..));
            }
        });
    }

    /// An asleep entry that has not been woken since `woken` was last emptied: `valid_orphan` would
    /// fail on it, or `is_idle_orphan` would hold, so the frame passes would skip it anyway.
    fn is_asleep(&self, entry: &OrphanEntry<'gc>) -> bool {
        entry.asleep
            && (self.woken.is_empty() || !self.woken.contains(&(entry.obj.as_ptr() as usize)))
    }

    /// Adds a `MovieClip` to the orphan list. In AVM2, movies advance their
    /// frames even when they are not on a display list. Unfortunately,
    /// multiple SWFS rely on this behavior, so we need to match Flash's
    /// behavior. This should not be called manually - `movie_clip` will
    /// call it when necessary.
    pub fn add_orphan_obj(&mut self, dobj: DisplayObject<'gc>) {
        // Note: comparing pointers is correct because GcWeak keeps its allocation alive,
        // so the pointers can't overlap by accident.
        if self.listed.insert(dobj.as_ptr() as usize) {
            self.orphans_mut().push(OrphanEntry {
                obj: dobj.downgrade(),
                asleep: false,
            });
        }
    }

    pub fn each_orphan_obj(
        context: &mut UpdateContext<'gc>,
        mut f: impl FnMut(DisplayObject<'gc>, &mut UpdateContext<'gc>),
    ) {
        // Clone the Rc before iterating over it. Any modifications must go through
        // `Rc::make_mut` in `orphan_objects_mut`, which will leave this `Rc` unmodified.
        // This ensures that any orphan additions/removals done by `f` will not affect
        // the iteration in this method.
        context.orphan_manager.drain_woken();
        let orphan_objs: Rc<_> = context.orphan_manager.orphans.clone();
        context.orphan_manager.passes += 1;

        for orphan in orphan_objs.iter() {
            if context.orphan_manager.is_asleep(orphan) {
                continue;
            }
            if let Some(dobj) = valid_orphan(orphan.obj, context.gc()) {
                f(dobj, context);
                // An orphan further down the list may have been woken by this one.
                context.orphan_manager.drain_woken();
            }
        }
        context.orphan_manager.passes -= 1;
    }

    /// Called at the end of `run_all_phases_avm2` - removes any movies
    /// that have been garbage collected, or are no longer orphans
    /// (they've since acquired a parent).
    pub fn cleanup_dead_orphans(&mut self, mc: &Mutation<'gc>) {
        self.drain_woken();
        let woken = std::mem::take(&mut self.woken);
        self.since_full_check += 1;
        let full_check = self.since_full_check >= 64;
        if full_check {
            self.since_full_check = 0;
        }
        self.orphans_mut().retain_mut(|e| {
            if e.asleep && !full_check && !woken.contains(&(e.obj.as_ptr() as usize)) {
                // Untouched since it was found idle: kept and still idle (perf-lab 0005). A dead
                // object stays in the list, skipped, until the next full check.
                return true;
            }
            if let Some(dobj) = valid_orphan(e.obj, mc) {
                // All clips that become orphaned (have their parent removed, or start out with no parent)
                // get added to the orphan list. However, there's a distinction between clips
                // that are removed from a RemoveObject tag, and clips that are removed from ActionScript.
                //
                // Clips removed from a RemoveObject tag only stay on the orphan list until the end
                // of the frame - this lets them run a framescript (with 'this.parent == null')
                // before they're removed. After that, they're removed from the orphan list,
                // and will not be run in any way.
                //
                // Clips removed from ActionScript stay on the orphan list, and will be run
                // indefinitely (if there are no remaining strong references, they will eventually
                // be garbage collected).
                //
                // To detect this, we check 'placed_by_avm2_script'. This flag get set to 'true'
                // for objects constructed from ActionScript, and for objects moved around
                // in the timeline (add/remove child, swap depths) by ActionScript. A
                // RemoveObject tag will only affect objects instantiated by the timeline,
                // which have not been moved in the displaylist by ActionScript. Therefore,
                // any orphan we see that has 'placed_by_avm2_script()' should stay on the orphan
                // list, because it was not removed by a RemoveObject tag.
                let keep = dobj.placed_by_avm2_script();
                // perf-lab 0005: an idle orphan falls asleep until something wakes it; 0006: the
                // same for a clip whose whole tree is idle, every object of the tree being marked.
                let idle = keep && dobj.as_movie_clip().is_some() && is_idle_tree(dobj);
                e.asleep = idle;
                if idle {
                    set_tree_asleep(dobj);
                } else {
                    dobj.base().set_orphan_asleep(false);
                }
                keep
            } else {
                // Dead, or inside another tree now: its flag may stay set, which can only cause
                // an extra wake-up (never a missed one).
                false
            }
        });
        self.listed.clear();
        self.listed
            .extend(self.orphans.iter().map(|e| e.obj.as_ptr() as usize));
        if self.passes > 0 {
            // A nested goto's cleanup, in the middle of a pass: the entries of its snapshot keep
            // their old `asleep`, so the wake-ups stay for it until the end-of-frame cleanup.
            self.woken = woken;
        }
    }
}

impl<'gc> Default for OrphanManager<'gc> {
    fn default() -> Self {
        Self {
            orphans: Rc::new(Vec::new()),
            listed: FnvHashSet::default(),
            woken: FnvHashSet::default(),
            since_full_check: 0,
            passes: 0,
        }
    }
}

/// If the provided `DisplayObjectWeak` should have frames run, returns
/// Some(clip) with an upgraded `MovieClip`.
/// If this returns `None`, the entry should be removed from the orphan list.
fn valid_orphan<'gc>(
    dobj: DisplayObjectWeak<'gc>,
    mc: &Mutation<'gc>,
) -> Option<DisplayObject<'gc>> {
    dobj.upgrade(mc).filter(|dobj| dobj.parent().is_none())
}
