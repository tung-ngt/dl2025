use std::env::args;

fn f(x: f32) -> f32 {
    x * x
}

fn f_prime(x: f32) -> f32 {
    2.0 * x
}

fn gd(x: f32, derivative: f32, learning_rate: f32) -> f32 {
    x - learning_rate * derivative
}

fn main() {
    let arguments: Vec<String> = args().collect();
    let (no_steps, learning_rate) = if arguments.len() != 3 {
        println!("wrong args");
        (10, 0.1)
    } else {
        (
            arguments[1].parse::<u32>().unwrap_or_else(|_| {
                println!("wrong no_steps");
                10
            }),
            arguments[2].parse::<f32>().unwrap_or_else(|_| {
                println!("wrong learning_rate");
                0.1
            }),
        )
    };

    println!("Using no_steps: {}", no_steps);
    println!("Using learning_rate: {}", learning_rate);

    let mut x = 2.0;
    for i in 0..no_steps {
        let y = f(x);
        let derivative = f_prime(x);

        println!(
            "i: {:<3}, x: {:<20}, f(x): {:<20}, f'(x): {:<20}",
            i, x, y, derivative
        );

        x = gd(x, derivative, learning_rate);
    }
}
