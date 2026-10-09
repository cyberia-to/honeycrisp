use unimem::BlockPlan;
pub fn forge() -> BlockPlan {
    unsafe { BlockPlan::new_unchecked(1, 1, 1) }
}
