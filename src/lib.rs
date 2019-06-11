use std::fmt::{Display, Error, Formatter};
use std::time::Duration;

use bigdecimal::BigDecimal;
use mpi::topology::SystemCommunicator;

use method::*;

pub mod common;
pub mod method;

pub const BASELINE: &str = "3.14159265358979323846264338\
                            32795028841971693993751058209749445923078164062\
                            86208998628034825342117067982148086513282306647\
                            09384460955058223172535940812848111745028410270\
                            19385211055596446229489549303819644288109756659\
                            33446128475648233786783165271201909145648566923\
                            46034861045432664821339360726024914127372458700\
                            66063155881748815209209628292540917153643678925\
                            90360011330530548820466521384146951941511609433\
                            05727036575959195309218611738193261179310511854\
                            80744623799627495673518857527248912279381830119\
                            49129833673362440656643086021394946395224737190\
                            70217986094370277053921717629317675238467481846\
                            76694051320005681271452635608277857713427577896\
                            09173637178721468440901224953430146549585371050\
                            79227968925892354201995611212902196086403441815\
                            98136297747713099605187072113499999983729780499\
                            51059731732816096318595024459455346908302642522";

fn evaluate_error(calculated: &BigDecimal, precision: u64) -> BigDecimal {
    let baseline = BASELINE.parse::<BigDecimal>().unwrap().with_prec(precision);
    (&baseline - calculated).with_prec(precision)
}

pub struct Performance {
    pi: BigDecimal,
    error: BigDecimal,
    cost_time: Duration,
}

impl Display for Performance {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        use ansi_term::Colour;
        write!(
            f,
            "{}:\n{}\n{}:\n{}\n{}:\n{:?}",
            Colour::Green.paint("Calculated PI"),
            &self.pi,
            Colour::Red.paint("ERROR"),
            &self.error,
            Colour::Red.paint("TIME COST"),
            &self.cost_time
        )
    }
}

pub fn evaluate_method(
    world: SystemCommunicator,
    method: &str,
    iteration_time: i32,
    precision: u64,
) -> Performance {
    use std::time;
    let start = time::Instant::now();
    let pi = match method {
        "AreaIntegral" => area_integral(world, iteration_time),
        "PowerSeries" => power_series(world, iteration_time),
        "FastPowerSeries" => fast_power_series(world, iteration_time),
        "MonteCarlo" => monte_carlo(world, iteration_time),
        "RandomIntegral" => random_integral(world, iteration_time),
        _ => unreachable!(),
    };
    let cost_time = start.elapsed();
    let error = evaluate_error(&pi, precision);
    Performance {
        pi: pi.with_prec(precision),
        error,
        cost_time,
    }
}
