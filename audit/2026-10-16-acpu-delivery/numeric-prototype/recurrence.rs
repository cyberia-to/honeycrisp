//! Audit prototype only: bounded scaled backward recurrence, no production imports.
#[derive(Clone, Copy)]
pub struct Limits {
    pub coefficients: usize,
    pub steps: u64,
}

pub const DEFAULT: Limits = Limits {
    coefficients: 65_536,
    steps: 1_000_000,
};

#[derive(Debug, PartialEq)]
pub enum Error {
    InvalidTime,
    OrderOverflow,
    OutputLimit,
    WorkLimit,
    Allocation,
    Convergence,
}

pub struct Output {
    pub coefficients: Vec<f32>,
    pub depth: usize,
    pub steps: u64,
    pub rounds: usize,
    pub tail_estimate: f64,
    pub largest_gap: f64,
}

struct Work {
    used: u64,
    limit: u64,
}

impl Work {
    fn charge(&mut self, steps: u64) -> Result<(), Error> {
        let next = self.used.checked_add(steps).ok_or(Error::WorkLimit)?;
        if next > self.limit {
            return Err(Error::WorkLimit);
        }
        self.used = next;
        Ok(())
    }
}

fn zeroed<T: Clone + Default>(len: usize) -> Result<Vec<T>, Error> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(len)
        .map_err(|_| Error::Allocation)?;
    result.resize(len, T::default());
    Ok(result)
}

// Chernoff estimate from the positive bilateral generating function. Algebra
// avoids cancellation in sqrt(1+u*u)-1. f64 transcendentals are not interval bounds.
fn log_tail(tau: f64, depth: usize) -> f64 {
    let n = depth as f64 + 1.0;
    let u = n / tau;
    std::f64::consts::LN_2 + n * (u / (1.0 + u.hypot(1.0)) - u.asinh())
}

pub fn coefficients(tau: f32, order: usize, limits: Limits) -> Result<Output, Error> {
    if !tau.is_finite() || tau < 0.0 {
        return Err(Error::InvalidTime);
    }
    let count = order.checked_add(1).ok_or(Error::OrderOverflow)?;
    if count > limits.coefficients {
        return Err(Error::OutputLimit);
    }
    let mut work = Work {
        used: 0,
        limit: limits.steps,
    };
    if tau == 0.0 {
        work.charge(count as u64)?;
        let mut result = zeroed(count)?;
        result[0] = 1.0;
        return Ok(Output {
            coefficients: result,
            depth: 0,
            steps: work.used,
            rounds: 0,
            tail_estimate: 0.0,
            largest_gap: 0.0,
        });
    }
    let mut depth = count.max(16);
    let tau = f64::from(tau);
    let mut tail;
    loop {
        work.charge(1)?;
        tail = log_tail(tau, depth);
        if !tail.is_finite() {
            return Err(Error::Convergence);
        }
        // No data allocation before the first unavoidable recurrence/output work fits.
        let required = (depth as u64)
            .checked_add(count as u64)
            .ok_or(Error::WorkLimit)?;
        if required > limits.steps - work.used {
            return Err(Error::WorkLimit);
        }
        if tail <= -40.0 * std::f64::consts::LN_2 {
            break;
        }
        depth = depth.checked_mul(2).ok_or(Error::WorkLimit)?;
    }
    let mut ratios: Vec<[f64; 2]> = zeroed(order)?;
    let mut result = zeroed(count)?;
    let mut previous_gap = f64::INFINITY;
    for rounds in 1.. {
        work.charge(
            (depth as u64)
                .checked_add(count as u64)
                .ok_or(Error::WorkLimit)?,
        )?;
        let mut ratio = [0.0, 1.0];
        let mut sum = [1.0, 1.0]; // sum of f[j..=N] / f[j]
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
                return Err(Error::Convergence);
            }
            let tolerance = high * 2.0_f64.powi(-25) + f64::from(f32::from_bits(1)) / 8.0;
            largest_gap = largest_gap.max((high - low) / tolerance);
            let middle = (low + high) * 0.5;
            result[k] = (if k % 2 == 0 { middle } else { -middle }) as f32;
        }
        if largest_gap <= 1.0 {
            return Ok(Output {
                coefficients: result,
                depth,
                steps: work.used,
                rounds,
                tail_estimate,
                largest_gap,
            });
        }
        if largest_gap >= previous_gap {
            return Err(Error::Convergence);
        }
        previous_gap = largest_gap;
        depth = depth.checked_mul(2).ok_or(Error::WorkLimit)?;
        work.charge(1)?;
        tail = log_tail(tau, depth);
    }
    unreachable!()
}
