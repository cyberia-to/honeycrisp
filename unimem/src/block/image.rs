use super::{aligned_size, checked_size, image_native, Block};
use crate::{ffi::*, MemError};

/// Native range and packing are explicit; no colour interpretation is inferred.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeImageFormat {
    Nv12VideoRange,
    Nv12FullRange,
    P010VideoRange,
    P010FullRange,
}
impl NativeImageFormat {
    pub fn fourcc(self) -> u32 {
        u32::from_be_bytes(match self {
            Self::Nv12VideoRange => *b"420v",
            Self::Nv12FullRange => *b"420f",
            Self::P010VideoRange => *b"x420",
            Self::P010FullRange => *b"xf20",
        })
    }
    pub(super) fn from_fourcc(value: u32) -> Result<Self, MemError> {
        [
            Self::Nv12VideoRange,
            Self::Nv12FullRange,
            Self::P010VideoRange,
            Self::P010FullRange,
        ]
        .into_iter()
        .find(|f| f.fourcc() == value)
        .ok_or(MemError::ImageFormatUnsupported(value))
    }
    pub(super) fn lane_bytes(self) -> usize {
        match self {
            Self::Nv12VideoRange | Self::Nv12FullRange => 1,
            _ => 2,
        }
    }
}

/// Checked native plane geometry in bytes, including complete final-row padding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativePlaneLayout {
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) element_bytes: usize,
    pub(super) stride: usize,
    pub(super) offset: usize,
    pub(super) row_extent: usize,
}
impl NativePlaneLayout {
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn element_bytes(&self) -> usize {
        self.element_bytes
    }
    pub fn stride(&self) -> usize {
        self.stride
    }
    pub fn offset(&self) -> usize {
        self.offset
    }
    /// Bytes through all full rows, including final-row padding. Additional
    /// native gaps/rounding belong to NativeImageLayout::allocation_size().
    pub fn row_extent(&self) -> usize {
        self.row_extent
    }
}

/// Validated numeric layout, with no initialization/access/readiness authority.
/// Its plane metadata cannot be rewritten after validation.
/// ```compile_fail,E0616
/// use unimem::{Block, NativeImageFormat};
/// let mut layout = *Block::plan_image(NativeImageFormat::Nv12VideoRange, 5, 3).unwrap().layout();
/// layout.planes[0].offset = usize::MAX;
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeImageLayout {
    pub(super) format: NativeImageFormat,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) allocation_size: usize,
    pub(super) planes: [NativePlaneLayout; 2],
}
impl NativeImageLayout {
    pub fn format(&self) -> NativeImageFormat {
        self.format
    }
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn allocation_size(&self) -> usize {
        self.allocation_size
    }
    pub fn planes(&self) -> &[NativePlaneLayout; 2] {
        &self.planes
    }
}

/// Allocation-free native image plan. Each open allocates an independent Block.
/// Reserve the complete extent plus caller controls before opening. Opening
/// does not initialize samples/padding or prove device readiness.
///
/// ```compile_fail,E0451
/// use unimem::{Block, ImageBlockPlan, NativeImageFormat};
/// let layout = *Block::plan_image(NativeImageFormat::Nv12VideoRange, 5, 3).unwrap().layout();
/// let forged = ImageBlockPlan { layout };
/// ```
/// ```compile_fail,E0599
/// let forged = unimem::ImageBlockPlan::new_unchecked(5, 3);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct ImageBlockPlan {
    layout: NativeImageLayout,
}
impl ImageBlockPlan {
    pub fn layout(&self) -> &NativeImageLayout {
        &self.layout
    }
    pub fn allocation_size(&self) -> usize {
        self.layout.allocation_size
    }
    pub fn open(&self) -> Result<Block, MemError> {
        image_native::open(self)
    }

    pub(super) fn checked(
        format: NativeImageFormat,
        width: usize,
        height: usize,
        row_alignment: usize,
        offset_alignment: usize,
        allocation_alignment: usize,
    ) -> Result<Self, MemError> {
        checked_size(width)?;
        checked_size(height)?;
        let lane = format.lane_bytes();
        let shapes = [
            (width, height, lane),
            (width / 2 + width % 2, height / 2 + height % 2, lane * 2),
        ];
        let mut end = 0;
        let mut planes = [NativePlaneLayout {
            width: 0,
            height: 0,
            element_bytes: 0,
            stride: 0,
            offset: 0,
            row_extent: 0,
        }; 2];
        for (plane, (w, h, bytes)) in planes.iter_mut().zip(shapes) {
            let row = w.checked_mul(bytes).ok_or(MemError::SizeOverflow)?;
            let stride = aligned_size(row, row_alignment)?;
            let size = stride.checked_mul(h).ok_or(MemError::SizeOverflow)?;
            checked_size(size)?;
            let offset = if end == 0 {
                0
            } else {
                aligned_size(end, offset_alignment)?
            };
            if !stride.is_multiple_of(lane) || !offset.is_multiple_of(lane) {
                return Err(MemError::BlockAlignmentInvalid);
            }
            end = offset.checked_add(size).ok_or(MemError::SizeOverflow)?;
            checked_size(end)?;
            *plane = NativePlaneLayout {
                width: w,
                height: h,
                element_bytes: bytes,
                stride,
                offset,
                row_extent: size,
            };
        }
        if offset_alignment == 0 {
            return Err(MemError::BlockAlignmentInvalid);
        }
        Ok(Self {
            layout: NativeImageLayout {
                format,
                width,
                height,
                allocation_size: aligned_size(end, allocation_alignment)?,
                planes,
            },
        })
    }
}

impl Block {
    /// Plan actual two-plane IOSurface allocation, without allocating CF/backing.
    pub fn plan_image(
        format: NativeImageFormat,
        width: usize,
        height: usize,
    ) -> Result<ImageBlockPlan, MemError> {
        // SAFETY: SDK keys are borrowed constants; native helpers consume only
        // representable checked numeric properties and allocate no backing.
        unsafe {
            let plan = ImageBlockPlan::checked(
                format,
                width,
                height,
                IOSurfaceGetPropertyAlignment(kIOSurfacePlaneBytesPerRow),
                IOSurfaceGetPropertyAlignment(kIOSurfacePlaneOffset),
                IOSurfaceGetPropertyAlignment(kIOSurfaceAllocSize),
            )?;
            for plane in plan.layout.planes {
                if IOSurfaceAlignProperty(
                    kIOSurfacePlaneBytesPerRow,
                    plane.width * plane.element_bytes,
                ) != plane.stride
                    || IOSurfaceAlignProperty(kIOSurfacePlaneOffset, plane.offset) != plane.offset
                {
                    return Err(MemError::BlockAlignmentInvalid);
                }
            }
            let last = plan.layout.planes[1];
            if IOSurfaceAlignProperty(kIOSurfaceAllocSize, last.offset + last.row_extent)
                != plan.allocation_size()
            {
                return Err(MemError::BlockAlignmentInvalid);
            }
            Ok(plan)
        }
    }

    /// Query and validate actual supported native geometry. This creates no
    /// sample reference and provides no initialization or access authority.
    pub fn image_layout(&self) -> Result<NativeImageLayout, MemError> {
        image_native::layout(self)
    }

    /// Retain an existing supported IOSurface and add one lifetime CPU lock.
    /// Valid padded native layouts are preserved. Caller ownership is unchanged.
    /// Null is rejected without accessing native memory.
    ///
    /// # Safety
    /// A non-null handle must be a live IOSurface for this call. Its geometry and
    /// mapping must remain stable through the returned Block's lifetime, including
    /// all other owners/imports. Retain establishes lifetime, not initialization,
    /// CPU/device exclusion or completion/visibility. Those remain the obligations
    /// of every unsafe sample view and native/raw use of the returned Block.
    ///
    /// ```compile_fail,E0133
    /// fn import(raw: unimem::ffi::IOSurfaceRef) { let _ = unimem::Block::retain_image(raw); }
    /// ```
    pub unsafe fn retain_image(surface: IOSurfaceRef) -> Result<Block, MemError> {
        // SAFETY: Forward the caller's live-handle and stable-layout contract.
        unsafe { image_native::retain(surface) }
    }
}
