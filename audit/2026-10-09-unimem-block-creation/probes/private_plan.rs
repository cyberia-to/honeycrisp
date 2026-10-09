use unimem::BlockPlan;
pub fn forge() -> BlockPlan {
    BlockPlan { requested: 1, row_bytes: 1, allocation_size: 1 }
}
