use super::*;

type G = Breakthrough;


#[test]
fn test_random_vs_flat_breaktrough() {
    let p1 = Random {};
    let p2 = FlatMonteCarlo {};

    assert!(monte_carlo_match::<G, _, _>(10, &p1, &p2) > 5)
}

#[test]
fn test_flat_vs_flatucb_breaktrough() {
    let p1 = FlatMonteCarlo {};
    let p2 = FlatUCBMonteCarlo {};

    assert!(monte_carlo_match::<G, _, _>(10, &p1, &p2) > 5)
}

#[test]
fn test_flatucb_vs_uct_breaktrough() {
    let p1 = FlatUCBMonteCarlo {};
    let p2 = UCT::default();

    assert!(monte_carlo_match::<G, _, _>(10, &p1, &p2) > 5)
}

#[test]
fn test_uct_vs_rave_breaktrough() {
    let p1 = UCT::default();
    let p2 = RAVE::default();

    assert!(monte_carlo_match::<G, _, _>(10, &p1, &p2) > 5)
}

