use mpi::topology::Communicator;
use mpi_pi::evaluate_method;
use structopt::*;

fn main() {
    let settings = Opt::from_args();
    let methods = [
        "AreaIntegral",
        "PowerSeries",
        // pass "FastPowerSeries",
        "MonteCarlo",
        "RandomIntegral",
    ];
    let universe = mpi::initialize().expect("Failed init mpi env");
    let world = universe.world();

    if settings.benchmark_all {
        let sp = std::iter::repeat("-").take(40).collect::<String>();
        methods.iter().for_each(|method| {
            let performance =
                evaluate_method(world, method, settings.iteration_time, settings.precision);
            if world.rank() == 0 {
                println!("Method:{}", method);
                println!("{}", &sp);
                println!("{}", performance);
            }
        });
        return;
    } else {
        let performance = evaluate_method(
            world,
            &settings.method,
            settings.iteration_time,
            settings.precision,
        );
        if world.rank() == 0 {
            println!("{}", performance);
        }
    }
}

#[derive(Debug, Clone, StructOpt)]
#[structopt(name = "Mpi-pi", about = "PI calculation using mpi", author = "sam")]
struct Opt {
    #[structopt(
        long,
        short,
        default_value = "AreaIntegral",
        raw(
            possible_values = r#"&["AreaIntegral","PowerSeries","FastPowerSeries","MonteCarlo","RandomIntegral"]"#
        )
    )]
    pub method: String,

    #[structopt(long, short)]
    pub benchmark_all: bool,
    #[structopt(
        name = "Iteration time",
        long = "iteration-time",
        short,
        default_value = "5000"
    )]
    pub iteration_time: i32,

    #[structopt(long, short, default_value = "500")]
    pub precision: u64,
}
