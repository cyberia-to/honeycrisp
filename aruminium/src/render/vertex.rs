//! Vertex descriptor — typed vertex input layout.
//!
//! Maps to `MTLVertexDescriptor`. Phase 2: required when a render pipeline
//! consumes typed `[[stage_in]]` vertex attributes rather than reading
//! raw buffer bytes via `device const T*`.

use crate::ffi::*;

/// Vertex attribute element format (matches `MTLVertexFormat`).
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub enum VertexFormat {
    Float = MTLVertexFormatFloat,
    Float2 = MTLVertexFormatFloat2,
    Float3 = MTLVertexFormatFloat3,
    Float4 = MTLVertexFormatFloat4,
    Half2 = MTLVertexFormatHalf2,
    Half4 = MTLVertexFormatHalf4,
    UChar4Normalized = MTLVertexFormatUChar4Normalized,
    UInt = MTLVertexFormatUInt,
    Int = MTLVertexFormatInt,
}

/// Step function (matches `MTLVertexStepFunction`).
#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub enum VertexStep {
    Constant = MTLVertexStepFunctionConstant,
    PerVertex = MTLVertexStepFunctionPerVertex,
    PerInstance = MTLVertexStepFunctionPerInstance,
}

/// One attribute (e.g. position, normal, uv) sourced from a buffer.
#[derive(Debug, Clone, Copy)]
pub struct VertexAttribute {
    /// Shader location ([[attribute(N)]]).
    pub shader_location: u32,
    pub format: VertexFormat,
    /// Byte offset within the buffer's stride.
    pub offset: u32,
    /// Which vertex buffer index this attribute reads from.
    pub buffer_index: u32,
}

/// Layout of one vertex buffer binding.
#[derive(Debug, Clone, Copy)]
pub struct VertexBufferLayout {
    /// Vertex buffer index ([[buffer(N)]]).
    pub buffer_index: u32,
    pub stride: u32,
    pub step: VertexStep,
    pub step_rate: u32,
}

/// Vertex descriptor — collection of attributes + layouts.
/// Pure data; converted to `MTLVertexDescriptor` at pipeline build time.
#[derive(Debug, Clone, Default)]
pub struct VertexDescriptor {
    pub attributes: Vec<VertexAttribute>,
    pub layouts: Vec<VertexBufferLayout>,
}

impl VertexDescriptor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_attribute(mut self, a: VertexAttribute) -> Self {
        self.attributes.push(a);
        self
    }

    pub fn with_layout(mut self, l: VertexBufferLayout) -> Self {
        self.layouts.push(l);
        self
    }

    /// Build an `id<MTLVertexDescriptor>`. Returns a +1 retained instance
    /// (via `+[MTLVertexDescriptor new]`).
    ///
    /// # Safety
    /// Caller owns the returned object and must release it (or hand it to
    /// a method that retains, e.g. `setVertexDescriptor:`).
    pub(crate) unsafe fn build_objc(&self) -> ObjcId {
        let cls = objc_getClass(c"MTLVertexDescriptor".as_ptr()) as ObjcId;
        let new_sel = sel_registerName(c"new".as_ptr());
        let desc = msg0(cls, new_sel);

        let attrs_array = msg0(desc, SEL_attributes());
        for a in &self.attributes {
            let slot = msg1_uint_id(
                attrs_array,
                SEL_objectAtIndexedSubscript(),
                a.shader_location as NSUInteger,
            );
            msg1_uint_void(slot, SEL_setFormat(), a.format as NSUInteger);
            msg1_uint_void(slot, SEL_setOffset(), a.offset as NSUInteger);
            msg1_uint_void(slot, SEL_setBufferIndex(), a.buffer_index as NSUInteger);
        }

        let layouts_array = msg0(desc, SEL_layouts());
        for l in &self.layouts {
            let slot = msg1_uint_id(
                layouts_array,
                SEL_objectAtIndexedSubscript(),
                l.buffer_index as NSUInteger,
            );
            msg1_uint_void(slot, SEL_setStride(), l.stride as NSUInteger);
            msg1_uint_void(slot, SEL_setStepFunction(), l.step as NSUInteger);
            msg1_uint_void(slot, SEL_setStepRate(), l.step_rate as NSUInteger);
        }
        desc
    }
}
