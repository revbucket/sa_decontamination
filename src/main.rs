


use std::cmp;
use std::collections::HashMap;
use serde_json::Value;
use std::io::BufRead;

use dashmap::{DashMap, DashSet};
use crate::io::{expand_dirs, read_pathbuf_to_mem, write_mem_to_pathbuf};
use crate::dedup::{load_sa_into_memory, get_occurrences_memory, load_size_object, doc_lookup};
use std::time::Instant;
use std::path::{PathBuf};
use anyhow::{Result, Error};
use rayon::prelude::*;
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use bincode;



pub mod s3;
pub mod io;
pub mod dedup;
pub mod table;



/*=================================================================
=                                 ARGS                            =
=================================================================*/


#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct ArgParser {
    #[clap(subcommand)]
    command: Commands,
}


#[derive(Subcommand, Debug)]
enum Commands {
    #[clap(arg_required_else_help = true)]
    BuildMatches { 
        #[arg(required=true, long)]
        data_file: PathBuf,

        #[arg(required=true, long, num_args=1..)]
        trainset: Vec<PathBuf>,

        #[arg(required=true, long)]
        output: PathBuf,

        #[arg(long, default_value_t=10)]
        match_size: usize
    },

    MarkContaminates {
        #[arg(required=true, long)]
        data_file: PathBuf, //used to infer where the size file lives

        #[arg(required=true, long)]
        match_location: PathBuf,

        #[arg(required=true, long)]
        output: PathBuf,

        #[arg(required=true, long)]
        threshold: f64,

        #[arg(required=true, long)]
        match_size: usize
    }


 }

/*=================================================================
=                              UTILITIES                          =
=================================================================*/

fn build_pbar(num_items: usize, units: &str) -> ProgressBar {
    let mut template = String::from(units);
    template.push_str(" {human_pos}/{human_len} [{elapsed_precise}/{duration_precise}] [{wide_bar:.cyan/blue}]");
    let pbar = ProgressBar::new(num_items as u64)
        .with_style(
            ProgressStyle::with_template(&template).unwrap()
        );
    pbar.inc(0);
    pbar
}




/*=================================================================
=                      MATCH BUILDER HELERS                       =
=================================================================*/

fn collect_matches(path: &PathBuf, path_idx: usize, text: &Vec<u8>, size_text: u64, table: &Vec<u8>, 
                   size_table: u64, size_width: usize, match_size: usize
                   ) -> Result<Vec<(usize, usize, u64)>, Error> {
    // Each document might match with format
    // (trainset_path_id, line_num, suffix_array_idx)

    let mut output: Vec<(usize, usize, u64)> = Vec::new();
    let data = read_pathbuf_to_mem(path).unwrap();

    for (line_num, line) in data.lines().enumerate() {
        let line = line.unwrap();
        let json: Value = serde_json::from_str(&line).unwrap();
        let line_text = json["text"].as_str().unwrap();        
        let line_text = line_text.as_bytes();
        // TODO, maybe use tokens^ ?        
        for query in line_text.windows(match_size) {
            for text_idx in get_occurrences_memory(text, size_text, table, size_table, query, size_width) {
                output.push((path_idx, line_num, text_idx));
            }
        }
    }
    Ok(output)
}



/*=================================================================
=                      MARK CONTAMINATES HELPERS                  =
=================================================================*/

fn merge_matches(val_doc_id: usize, doc_matches: &DashMap<(usize, usize), Vec<u64>>, match_size: usize,
                 val_doc_size: usize, threshold: f64) -> Vec<(usize, usize, usize)> {
    // Groups into a vec of (val_doc_id, trainset_path_id, line_num)
    // For any trainset docs that surpass the threshold
    let mut output: Vec<(usize, usize, usize)> = Vec::new();
    
    doc_matches.iter()
        .for_each(|entry| {
            let (train_path_id, line_num) = *entry.key();
            if _check_threshold(entry.value(), match_size, val_doc_size, threshold) {
                output.push((val_doc_id, train_path_id, line_num))
            }
        });

    output
}


fn _check_threshold(interval_starts: &Vec<u64>, match_size: usize, doc_size: usize, threshold: f64) -> bool {
    // Checks if the matches of size match_size starting at interval_starts cover >= threshold * doc_size bytes
    let usize_threshold = ((doc_size as f64) * threshold).ceil() as usize;
    let intervals: Vec<(usize, usize)> = interval_starts.iter()
        .map(|start| (*start as usize, *start as usize +match_size))
        .collect();
    let merged_intervals = _merge_intervals(intervals, false);
    let total_width: usize = merged_intervals.iter().map(|(s, e)| e -s).sum();
    total_width >= usize_threshold
}


fn _merge_intervals(mut v: Vec<(usize, usize)>, already_sorted: bool) -> Vec<(usize, usize)>{    
    if !already_sorted {
        v.sort_by_key(|(key, _)| key.clone());
    }
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (s, e) in v {
        if merged.len() == 0 {
            merged.push((s, e));
        } else if merged.last().unwrap().1 >= s {
            let (old_s, old_e) = merged.pop().unwrap();
            merged.push((old_s, cmp::max(e, old_e)));
        } else {
            merged.push((s, e));
        }
    }
    merged
}

/*=================================================================
=                             Subcommands                         =
=================================================================*/



fn build_matches(data_file: &PathBuf, trainset: &Vec<PathBuf>, output: &PathBuf, match_size: usize) -> Result<(), Error> {
    println!("Starting Match Building run...");    
    let start_main = Instant::now();
    // Phase 0: Setup, collect filenames, build path lookup, build band seeds
    let mut input_files = expand_dirs(trainset.clone(), None).unwrap();
    input_files.sort(); // sort before building the path lookup
    let path_map : HashMap<PathBuf, usize> = input_files.iter()
        .enumerate()
        .map(|(index, path)| (path.clone(), index))
        .collect();

    println!("Collected {:?} input files", input_files.len());
    let (text, size_text, table, size_table, size_width) = load_sa_into_memory(data_file);

    // Phase 1: Collect all matches
    println!("Starting match collection...");
    let match_start = Instant::now();
    let pbar = build_pbar(input_files.len(), "Paths");
    let matches: Vec<(usize, usize, u64)> = path_map.par_iter()
        .flat_map(|(p, idx)| {
            let matches = collect_matches(p, *idx, &text, size_text, &table, size_table, size_width, match_size).unwrap();
            pbar.inc(1);
            matches
            })        
        .collect();
    println!("Collected {:?} matches", matches.len());
    println!("Match collection copleted in {:?} secs", match_start.elapsed().as_secs());

    // Phase 2: Save everything
    let path_map_json_bytes: Vec<u8> = serde_json::to_vec(&path_map).unwrap();
    write_mem_to_pathbuf(&path_map_json_bytes, &output.clone().join("paths.json.gz")).unwrap();
    let serialized_matches: Vec<u8> = bincode::serialize(&matches).unwrap();
    write_mem_to_pathbuf(&serialized_matches, &output.clone().join("matches.bin.gz")).unwrap();

    // Phase 3, finish up
    println!("-------------------------");
    println!("Completing match collection");
    println!("Found {:?} matches from {:?} paths", matches.len(), input_files.len());
    println!("Total runtime: {:?} secs", start_main.elapsed().as_secs());
    Ok(())
}


fn mark_contaminates(data_file: &PathBuf, match_location: &PathBuf, output: &PathBuf, threshold: f64, match_size: usize) -> Result<(), Error> {

    println!("Starting contaminate marking...");
    let start_main = Instant::now();
    // Phase 0: Load everything into mem
    let match_data_bytes = read_pathbuf_to_mem(match_location).unwrap().into_inner().into_inner();
    let matches: Vec<(usize, usize, u64)> = bincode::deserialize(&match_data_bytes).unwrap();
    let size_object_path = data_file.clone().join(".size");
    let size_object = load_size_object(&size_object_path);

    // Phase 1: group all matches by their val set id (and do path lookups)
    println!("Starting grouping of matches...");
    let start_group = Instant::now();
    let match_groups: DashMap<(usize, usize), DashMap<(usize, usize), Vec<u64>>> = DashMap::new();
    // Match groups maps:
    // {(Val_set_doc_id, Val_set_doc_len) -> 
    //            {train_set_doc_id -> [in_doc_pos]}
    // }
    let pbar = build_pbar(matches.len(), "Matches");
    matches.into_par_iter()
        .for_each(|(path_id, line_num, sa_pos)| {
            let val_doc_id = doc_lookup(sa_pos, &size_object);
            let in_doc_pos = sa_pos - size_object[val_doc_id];
            let val_doc_size = size_object[val_doc_id+1] - size_object[val_doc_id]; // MINUS NAME HERE???
            match_groups.entry((val_doc_id, val_doc_size.try_into().unwrap())).or_default()
                .entry((path_id, line_num)).or_default()
                .push(in_doc_pos);
            pbar.inc(1);
        });
    println!("Grouped matches in {:?} secs", start_group.elapsed().as_secs());

    // Phase 2: For each group merge intervals and compute thresholds
    println!("Starting contaminate aggregation...");
    let merge_start = Instant::now();
    let pbar = build_pbar(match_groups.len(), "Groups");

    let contaminates: Vec<(usize, usize, usize)> = match_groups.iter().par_bridge().flat_map(|entry| {
        let (val_doc_id, val_doc_size) = *entry.key();
        let merged_matches = merge_matches(val_doc_id, entry.value(), match_size, val_doc_size, threshold);
        pbar.inc(1);
        merged_matches})
        .collect();
    println!("Finishing aggregating contaminates in {:?} secs", merge_start.elapsed().as_secs());

    // Phase 3: Save contaminates
    let contaminate_bytes = bincode::serialize(&contaminates).unwrap();
    write_mem_to_pathbuf(&contaminate_bytes, &output.clone().join("contaminates.bin.gz")).unwrap();

    // Phase 4: Finalize
    let total_contams: DashSet<usize> = DashSet::new();
    contaminates.par_iter()
        .for_each(|(val_doc_id, _, _)| {
            total_contams.insert(*val_doc_id);
    });
    println!("-------------------------");
    println!("Completing contaminate collection");
    println!("Found {:?} contaminated val set docs", total_contams.len());
    println!("Found {:?} total contaminates", contaminates.len());
    println!("Total runtime: {:?} secs", start_main.elapsed().as_secs());
    Ok(())
}


/*=================================================================
=                                 MAIN                            =
=================================================================*/

fn main() {
    let args = ArgParser::parse();

    let result = match &args.command {
        Commands::BuildMatches {data_file, trainset, output, match_size} => {
            build_matches(data_file, trainset, output, *match_size)
        },        
        Commands::MarkContaminates {data_file, match_location, output, threshold, match_size} => {
            mark_contaminates(data_file, match_location, output, *threshold, *match_size)
        }
    };
    result.unwrap()
}


