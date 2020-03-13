
use tokio;

fn main() {
    let mut threaded_rt = tokio::runtime::Builder::new()
        .threaded_scheduler()
        .core_threads(8)
        .build().unwrap();

    threaded_rt.block_on(zerol::misc::game_generator())
}