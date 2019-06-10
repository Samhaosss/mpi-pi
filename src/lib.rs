use bigdecimal::BigDecimal;
use mpi;
use mpi::topology::{Rank, SystemCommunicator};

use common::*;

mod common;

fn f(x: BigDecimal) -> BigDecimal {
    BigDecimal::from(4.0) / (x.square() + BigDecimal::from(1.0))
}

pub fn area_integral(
    world: SystemCommunicator,
    rank: Rank,
    size: Rank,
    iteration_time: u64,
) -> BigDecimal {
    let h = BigDecimal::from(iteration_time).inverse();
    let pi = (1..iteration_time)
        .step_by(size as usize)
        .map(BigDecimal::from)
        .fold(BigDecimal::from(0.0), |x, y| {
            x + f(&h * (y - BigDecimal::from(0.5)))
        })
        * h;
    reduce_sum_big_decimal(world, rank, size, pi)
}

pub fn power_series() {
    unimplemented!()
}

pub fn fast_power_series() {
    unimplemented!()
}

pub fn monte_carlo() {
    unimplemented!()
}

pub fn random_integral() {
    unimplemented!()
}
