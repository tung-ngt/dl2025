fn f(x: f32) -> f32 {
    x * x
}

fn f_prime(x: f32) -> f32 {
    2.0 * x
}

fn sgd(x: f32, derivative: f32, learning_rate: f32) -> f32 {
    x - learning_rate * derivative
}

fn main() {
    let no_steps = 20;
    let learning_rate = 0.1;

    let mut x = 2.0;
    for i in 0..no_steps {
        let y = f(x);
        let derivative = f_prime(x);

        println!(
            "i: {}, x: {:<20}, f(x): {:<20}, f'(x): {:<20}",
            i, x, y, derivative
        );

        x = sgd(x, derivative, learning_rate);
    }
}
