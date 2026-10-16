//! Bounded scaled Bessel coefficients for the Chebyshev heat expansion.

/// Output and work budgets for coefficient generation; matvec work is separate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChebyshevLimits {
    /// Maximum number of returned coefficients, including the zeroth.
    pub max_coefficients: usize,
    /// Maximum tail probes, paired backward indices and coefficient evaluations.
    /// Initialization additionally touches at most twice the output count minus one.
    pub max_steps: u64,
}

/// Finite defaults used by [`super::chebyshev::chebyshev_coeffs`].
pub const DEFAULT_CHEBYSHEV_LIMITS: ChebyshevLimits = ChebyshevLimits {
    max_coefficients: 65_536,
    max_steps: 1_000_000,
};

impl Default for ChebyshevLimits {
    fn default() -> Self {
        DEFAULT_CHEBYSHEV_LIMITS
    }
}

/// A refusal before publishing coefficients or writing matvec output/scratch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChebyshevError {
    /// Time must be finite and nonnegative.
    InvalidTime,
    /// The requested order plus one is unrepresentable.
    OrderOverflow,
    /// The output count exceeds its caller-provided limit.
    OutputLimit,
    /// The required coefficient work exceeds its caller-provided limit.
    WorkLimit,
    /// A vector layout is unrepresentable or allocation was refused.
    Allocation,
    /// Numerical estimates are invalid or refinement fails to reduce their width.
    Convergence,
}

impl core::fmt::Display for ChebyshevError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::InvalidTime => "time must be finite and nonnegative",
            Self::OrderOverflow => "order plus one overflows",
            Self::OutputLimit => "coefficient count exceeds the output limit",
            Self::WorkLimit => "coefficient work exceeds the step limit",
            Self::Allocation => "coefficient allocation or layout was refused",
            Self::Convergence => "coefficient estimates did not converge",
        })
    }
}

impl std::error::Error for ChebyshevError {}

struct Work {
    used: u64,
    limit: u64,
}

impl Work {
    fn charge(&mut self, steps: u64) -> Result<(), ChebyshevError> {
        let next = self
            .used
            .checked_add(steps)
            .ok_or(ChebyshevError::WorkLimit)?;
        if next > self.limit {
            return Err(ChebyshevError::WorkLimit);
        }
        self.used = next;
        Ok(())
    }
}

fn zeroed<T: Clone + Default>(len: usize) -> Result<Vec<T>, ChebyshevError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(len)
        .map_err(|_| ChebyshevError::Allocation)?;
    result.resize(len, T::default());
    Ok(result)
}

// DLMF10.35.1 gives a Chernoff bound on the omitted positive normalization
// mass. This stable algebra avoids sqrt(1+u*u)-1 cancellation; f64 evaluation
// of it and the endpoint recurrences supplies estimates, not interval proofs.
fn log_tail(tau: f64, depth: usize) -> f64 {
    let n = depth as f64 + 1.0;
    let u = n / tau;
    std::f64::consts::LN_2 + n * (u / (1.0 + u.hypot(1.0)) - u.asinh())
}

/// Compute scaled heat coefficients with explicit output and work limits.
///
/// Returns `order + 1` coefficients: `exp(-tau) I_0(tau)` at zero and
/// `2 (-1)^k exp(-tau) I_k(tau)` above zero. Zero time returns `[1, 0, ...]`.
/// Requested heap elements occupy `16*order + 4*(order+1)` bytes; allocator
/// overhead is additional. No retained buffer grows with recurrence depth.
///
/// The two terminal-ratio sequences and estimated normalization tail provide
/// a convergence criterion, not a certified global accuracy bound. Requested
/// truncation order and matrix spectral assumptions are separate concerns.
///
/// # Errors
/// Returns [`ChebyshevError`] for invalid input, exhausted output/work budgets,
/// allocation/layout refusal or numerical nonconvergence. Refinements count
/// toward the same work budget. Callers can raise limits explicitly.
pub fn try_chebyshev_coeffs(
    tau: f32,
    order: usize,
    limits: ChebyshevLimits,
) -> Result<Vec<f32>, ChebyshevError> {
    if !tau.is_finite() || tau < 0.0 {
        return Err(ChebyshevError::InvalidTime);
    }
    let count = order.checked_add(1).ok_or(ChebyshevError::OrderOverflow)?;
    if count > limits.max_coefficients {
        return Err(ChebyshevError::OutputLimit);
    }
    let mut work = Work {
        used: 0,
        limit: limits.max_steps,
    };
    if tau == 0.0 {
        work.charge(count as u64)?;
        let mut result = zeroed(count)?;
        result[0] = 1.0;
        return Ok(result);
    }
    let mut depth = count.max(16);
    let tau = f64::from(tau);
    let mut tail;
    loop {
        work.charge(1)?;
        tail = log_tail(tau, depth);
        if !tail.is_finite() {
            return Err(ChebyshevError::Convergence);
        }
        let required = (depth as u64)
            .checked_add(count as u64)
            .ok_or(ChebyshevError::WorkLimit)?;
        if required > limits.max_steps - work.used {
            return Err(ChebyshevError::WorkLimit);
        }
        if tail <= -40.0 * std::f64::consts::LN_2 {
            break;
        }
        depth = depth.checked_mul(2).ok_or(ChebyshevError::WorkLimit)?;
    }
    let mut ratios: Vec<[f64; 2]> = zeroed(order)?;
    let mut result = zeroed(count)?;
    let mut previous_gap = f64::INFINITY;
    loop {
        work.charge(
            (depth as u64)
                .checked_add(count as u64)
                .ok_or(ChebyshevError::WorkLimit)?,
        )?;
        // I[N+1]/I[N] is between zero and one (DLMF10.37). In exact
        // arithmetic each normalized component is linear-fractional in that
        // terminal ratio, so the two endpoints bracket its truncated value.
        let mut ratio = [0.0, 1.0];
        let mut sum = [1.0, 1.0];
        for j in (1..=depth).rev() {
            let term = 2.0 * j as f64 / tau;
            for side in 0..2 {
                ratio[side] = 1.0 / (term + ratio[side]);
                sum[side] = 1.0 + ratio[side] * sum[side];
            }
            if j <= order {
                ratios[j - 1] = ratio;
            }
        }
        let mut q = [1.0 / (2.0 * sum[0] - 1.0), 1.0 / (2.0 * sum[1] - 1.0)];
        let tail_estimate = tail.exp();
        let mut largest_gap = 0.0_f64;
        for k in 0..count {
            if k != 0 {
                q[0] *= ratios[k - 1][0];
                q[1] *= ratios[k - 1][1];
            }
            let factor = if k == 0 { 1.0 } else { 2.0 };
            let low = q[0].min(q[1]) * (1.0 - tail_estimate) * factor;
            let high = q[0].max(q[1]) * factor;
            if !low.is_finite() || !high.is_finite() || low < 0.0 || high < low {
                return Err(ChebyshevError::Convergence);
            }
            let tolerance = high * 2.0_f64.powi(-25) + f64::from(f32::from_bits(1)) / 8.0;
            largest_gap = largest_gap.max((high - low) / tolerance);
            let middle = (low + high) * 0.5;
            result[k] = (if k % 2 == 0 { middle } else { -middle }) as f32;
        }
        if largest_gap <= 1.0 {
            return Ok(result);
        }
        if largest_gap >= previous_gap {
            return Err(ChebyshevError::Convergence);
        }
        previous_gap = largest_gap;
        depth = depth.checked_mul(2).ok_or(ChebyshevError::WorkLimit)?;
        work.charge(1)?;
        tail = log_tail(tau, depth);
    }
}
