//! Read an initial edge list and one delta edge list, run CommunityState::initial and ::update,
//! and print (as tab-separated lines) everything needed to validate the output in Python:
//! per-level node->community vectors before and after the update, the flat partition and its modularity.
use hit_leiden::{CommunityState, GraphInput, RunConfig};
use std::io::Write;

fn read(path: &str) -> (usize, Vec<(usize, usize, Option<f64>)>) {
    let s = std::fs::read_to_string(path).unwrap();
    let mut it = s.lines();
    let n: usize = it.next().unwrap().trim().parse().unwrap();
    let e = it.filter(|l| !l.is_empty()).map(|l| {
        let mut p = l.split_whitespace();
        (p.next().unwrap().parse().unwrap(), p.next().unwrap().parse().unwrap(), Some(1.0))
    }).collect();
    (n, e)
}

fn dump(out: &mut impl Write, tag: &str, v: &[usize]) {
    let s: Vec<String> = v.iter().map(|x| x.to_string()).collect();
    writeln!(out, "{}\t{}", tag, s.join(" ")).unwrap();
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (n0, e0) = read(&a[1]);
    let (n1, e1) = read(&a[2]);
    let cfg = RunConfig::default();
    let t = std::time::Instant::now();
    let (st, p0) = CommunityState::initial(&GraphInput { dataset_id: None, node_count: n0, edges: e0 }, &cfg).unwrap();
    let t_init = t.elapsed().as_secs_f64();
    let lv0: Vec<Vec<usize>> = st.hierarchy_levels().to_vec();
    let t = std::time::Instant::now();
    let (st2, p1) = st.update(&GraphInput { dataset_id: None, node_count: n1, edges: e1 }, &cfg).unwrap();
    let t_upd = t.elapsed().as_secs_f64();
    let mut o = std::io::BufWriter::new(std::io::stdout());
    writeln!(o, "meta\t{}\t{}\t{}\t{}\t{}\t{}", p0.quality_score, p1.quality_score, t_init, t_upd, lv0.len(), st2.hierarchy_levels().len()).unwrap();
    dump(&mut o, "flat0", &p0.node_to_community);
    dump(&mut o, "flat1", &p1.node_to_community);
    for (i, l) in lv0.iter().enumerate() { dump(&mut o, &format!("lv0_{}", i), l); }
    for (i, l) in st2.hierarchy_levels().iter().enumerate() { dump(&mut o, &format!("lv1_{}", i), l); }
}
