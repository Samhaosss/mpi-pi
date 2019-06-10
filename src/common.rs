use bigdecimal::BigDecimal;
use mpi;
use mpi::point_to_point::{Destination, Source, Status};
use mpi::topology::{Communicator, Rank, SystemCommunicator};

const ROOT: i32 = 0;

pub fn send_big_decimal(world: SystemCommunicator, destination: Rank, number: &BigDecimal) {
    let number_str = number.to_string();
    world
        .process_at_rank(destination)
        .send(number_str.as_bytes());
}

pub fn receive_big_decimal(world: SystemCommunicator, from: Rank) -> (BigDecimal, Status) {
    let (msg, status): (Vec<u8>, _) = world.process_at_rank(from).receive_vec();
    let number_str = String::from_utf8(msg).unwrap();
    (number_str.parse::<BigDecimal>().unwrap(), status)
}

pub fn reduce_sum_big_decimal(
    world: SystemCommunicator,
    rank: Rank,
    size: Rank,
    number: BigDecimal,
) -> BigDecimal {
    if rank == ROOT {
        (rank + 1..size)
            .map(|rank| receive_big_decimal(world, rank).0)
            .fold(number, |x, y| x + y)
    } else {
        send_big_decimal(world, ROOT, &number);
        number
    }
}
