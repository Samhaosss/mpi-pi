use bigdecimal::*;
use mpi::collective::{Root, SystemOperation};
use mpi::topology::{Communicator, SystemCommunicator};
use rand;

use crate::common::*;

pub fn area_integral(world: SystemCommunicator, iteration_time: i32) -> BigDecimal {
    let h = 1.0 / f64::from(iteration_time);
    let pi = (1..iteration_time)
        .step_by(world.size() as usize)
        .map(|n| {
            (BigDecimal::from(h * (f64::from(n) as f64 - 0.5)).square() + BigDecimal::from(1))
                .inverse()
        })
        .sum::<BigDecimal>()
        * BigDecimal::from(h)
        * BigDecimal::from(4.0);
    reduce_big_decimal(world, pi)
}

pub fn power_series(world: SystemCommunicator, iteration_time: i32) -> BigDecimal {
    let sub_pi = (world.rank()..iteration_time)
        .step_by(world.size() as usize)
        .map(|n| {
            let p = if n % 2 == 0 { 1 } else { -1 };
            BigDecimal::from((2 * n + 1) * p).inverse()
        })
        .sum::<BigDecimal>();
    reduce_big_decimal(world, sub_pi) * BigDecimal::from(4)
}

pub fn fast_power_series(_world: SystemCommunicator, _iteration_time: i32) -> BigDecimal {
    let first_part = (0..4)
        .map(|x| {
            let i = if x % 2 == 0 { 1 } else { -1 };
            // should not panic here
            (BigDecimal::from((x * 2 + 1) * i) * BigDecimal::from(5u64.pow((x * 2 + 1) as u32)))
                .inverse()
        })
        .sum::<BigDecimal>()
        * BigDecimal::from(4);
    let second_part = (0..4)
        .map(|x| {
            let i = if x % 2 == 0 { 1 } else { -1 };
            // should not panic here
            (BigDecimal::from((x * 2 + 1) * i) * BigDecimal::from(239u64.pow((x * 2 + 1) as u32)))
                .inverse()
        })
        .sum::<BigDecimal>()
        * BigDecimal::from(4);
    first_part - second_part
}

pub fn monte_carlo(world: SystemCommunicator, iteration_time: i32) -> BigDecimal {
    let local_count = (world.rank()..iteration_time)
        .step_by(world.size() as usize)
        .fold(0i32, |count, _| {
            let (x, y): (f64, f64) = (rand::random(), rand::random());
            if x * x + y * y <= 1.0 {
                count + 1
            } else {
                count
            }
        });
    let sum = if world.rank() == 0 {
        let mut sum = 0;
        world
            .process_at_rank(0)
            .reduce_into_root(&local_count, &mut sum, SystemOperation::sum());
        sum
    } else {
        world
            .process_at_rank(0)
            .reduce_into(&local_count, SystemOperation::sum());
        local_count
    };
    BigDecimal::from(4) * BigDecimal::from(sum) / BigDecimal::from(iteration_time)
}

pub fn random_integral(world: SystemCommunicator, iteration_time: i32) -> BigDecimal {
    let sub_pi = (world.rank()..iteration_time)
        .step_by(world.size() as usize)
        .map(|_n| {
            let x: f64 = rand::random();
            (BigDecimal::from(x).square() + BigDecimal::from(1)).inverse()
        })
        .sum::<BigDecimal>();
    reduce_big_decimal(world, sub_pi) * BigDecimal::from(4.0)
}
