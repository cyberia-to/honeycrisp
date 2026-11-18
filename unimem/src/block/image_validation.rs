use super::{checked_size, image::*};
use crate::MemError;

#[derive(Clone, Copy)]
pub(super) struct PlaneQuery {
    pub width: usize,
    pub height: usize,
    pub element_bytes: usize,
    pub element_width: usize,
    pub element_height: usize,
    pub stride: usize,
    pub base: usize,
}
#[derive(Clone, Copy)]
pub(super) struct ImageQuery {
    pub format: u32,
    pub width: usize,
    pub height: usize,
    pub count: usize,
    pub base: usize,
    pub allocation_size: usize,
    pub planes: [PlaneQuery; 2],
}

pub(super) fn validate(query: ImageQuery) -> Result<NativeImageLayout, MemError> {
    let invalid = || MemError::ImageLayoutInvalid;
    let format = NativeImageFormat::from_fourcc(query.format)?;
    checked_size(query.width)?;
    checked_size(query.height)?;
    checked_size(query.allocation_size)?;
    if query.count != 2 {
        return Err(invalid());
    }
    let lane = format.lane_bytes();
    if query.base == 0
        || !query.base.is_multiple_of(core::mem::align_of::<f32>())
        || query.base.checked_add(query.allocation_size).is_none()
    {
        return Err(MemError::BlockAddressInvalid);
    }
    let shapes = [
        (query.width, query.height, lane),
        (
            query.width / 2 + query.width % 2,
            query.height / 2 + query.height % 2,
            lane * 2,
        ),
    ];
    let mut planes = [NativePlaneLayout {
        width: 0,
        height: 0,
        element_bytes: 0,
        stride: 0,
        offset: 0,
        row_extent: 0,
    }; 2];
    for ((plane, actual), (width, height, bytes)) in planes.iter_mut().zip(query.planes).zip(shapes)
    {
        if (
            actual.width,
            actual.height,
            actual.element_bytes,
            actual.element_width,
            actual.element_height,
        ) != (width, height, bytes, 1, 1)
        {
            return Err(invalid());
        }
        let row = width.checked_mul(bytes).ok_or(MemError::SizeOverflow)?;
        let offset = actual.base.checked_sub(query.base).ok_or_else(invalid)?;
        if actual.stride < row
            || !actual.stride.is_multiple_of(lane)
            || !offset.is_multiple_of(lane)
        {
            return Err(invalid());
        }
        let size = actual
            .stride
            .checked_mul(height)
            .ok_or(MemError::SizeOverflow)?;
        let end = offset.checked_add(size).ok_or(MemError::SizeOverflow)?;
        if end > query.allocation_size {
            return Err(invalid());
        }
        *plane = NativePlaneLayout {
            width,
            height,
            element_bytes: bytes,
            stride: actual.stride,
            offset,
            row_extent: size,
        };
    }
    let [first, second] = planes;
    if first.offset < second.offset + second.row_extent
        && second.offset < first.offset + first.row_extent
    {
        return Err(invalid());
    }
    Ok(NativeImageLayout {
        format,
        width: query.width,
        height: query.height,
        allocation_size: query.allocation_size,
        planes,
    })
}
