# =============================================================================
# METABOLITE NETWORKING — MAIN DRIVER
#
# Goal: within each synthetic-liver sample (1-2 co-dosed drugs), hypothesize
# parent-drug -> metabolite relationships from MS1 mass differences, then
# confirm/score each hypothesis with two independent MS2-based match factors.
# Nothing in this pipeline compares against any external library — every
# comparison is within-sample, feature vs. feature, as discussed.
#
# Pipeline:
#   1. MS1 layer — for every within-sample feature pair, test whether the
#      mass difference matches a curated biotransformation Δm
#      (biotransformation_deltas.csv) -> candidate parent/metabolite edges.
#      No RT filter is applied (metabolites aren't reliably earlier/later
#      eluting than their parent).
#   2. MS2 layer — for each candidate edge only (not all-vs-all), compute the
#      combined entropy+cosine+tanimoto match factor from 3_Functions.jl:
#        (a) on the raw fragment spectra
#        (b) on CNL spectra built from your ORIGINAL fixed CNL bins
#            (CNL Headers.csv), kept intensity-weighted rather than binary —
#            see frags_to_cnl_spectrum's docstring for why.
#   3. Output — one edge list per sample (edges_<sample>.csv), ready to load
#      into a network tool (Graphs.jl/GraphMakie, or export straight into
#      Cytoscape/Gephi as an edge list) and filter/threshold however you like.
#
# BEFORE RUNNING — things specific to your data that you should sanity-check:
#   - Column names: this assumes MeasMass (precursor mass) and MS2Comp
#     (fragment m/z, JSON-array-as-string) per your original sketch. The
#     fragment INTENSITY column name is a guess (:MS2CompInt) — see the
#     warning build_feature_lookup will print if that's wrong, and the
#     docstring in metabolite_networking_functions.jl for how to fix it.
#   - ppm_tol / cnl_tol / ms2_da defaults below are reasonable starting
#     points for HRMS, not tuned to your specific instrument — adjust freely.
# =============================================================================

using DataFrames, CSV, JSON3
using ProgressMeter, Statistics

include("3_Functions.jl")                     # SpectrumEntry, match_peaks_in_spectra,
                                               # _pairwise_similarity, parse_mz, etc.
                                               # (unmodified — nothing here touches it)
include("metabolite_networking_functions.jl") # everything new; see that file's
                                               # top-of-file comment for a map of
                                               # what each function does

"""Parse a JSON-formatted string of floats, returning Float64[] on failure.
(unchanged from your original sketch)"""
function safe_parse_array(x)
    try
        return JSON3.read(replace(x, " " => ""), Vector{Float64})
    catch
        return Float64[]
    end
end

# ── Load reference tables (shared across every sample) ──────────────────────
cnl_headers = Float64.(CSV.read("CNL Headers.csv", DataFrame)[:, 1])
delta_table = load_delta_table("biotransformation_deltas (1).csv")

println("Loaded $(length(cnl_headers)) CNL bins and $(nrow(delta_table)) curated biotransformations.")

# ── Loop over samples ────────────────────────────────────────────────────────
all_CompCreate = "CompCreate data/"
all_files      = readdir(all_CompCreate)

for fname in all_files
    println("\n=== Processing $fname ===")
    df = CSV.read(joinpath(all_CompCreate, fname), DataFrame)

    # Every feature needs a stable ID to key the lookup dict and the output
    # edges by. If your CompCreate files already have a proper feature/compound
    # ID column, use it here instead (e.g. df.feature_id = df.CompoundID) so IDs
    # are meaningful across re-runs rather than just row position.
    if !hasproperty(df, :feature_id)
        df[!, :feature_id] = 1:nrow(df)
    end

    # ── Step 1: MS1 candidate edges ──────────────────────────────────────────
    # AccuMass is used here (not MeasMass) since it's already adduct/charge-
    # corrected — comparable across features even if they carry different
    # adducts, and consistent with the neutral-formula Δm values in the delta
    # table. MeasMass is intentionally kept for CNL generation further down,
    # since neutral losses need the actual detected ion masses, not the
    # adduct-corrected ones.
    edges = find_ms1_candidate_edges(
        Float64.(df.AccuMass), df.feature_id, delta_table;
        ppm_tol = 2.5,   # adjust to your instrument's actual mass accuracy
    )
    println("  MS1 candidate edges: $(nrow(edges))")

    if nrow(edges) == 0
        println("  No candidate edges for this sample — skipping MS2 scoring.")
        continue
    end

    # ── Step 2: MS2 + CNL confirmation ───────────────────────────────────────
    lookup = build_feature_lookup(df; id_col=:feature_id)
    scored = score_candidate_edges(edges, lookup, cnl_headers)

    # ── Step 3: save the full (unfiltered) scored edge table ────────────────
    out_name = "edges_" * splitext(fname)[1] * ".csv"
    CSV.write(out_name, scored)
    println("  Saved $out_name ($(nrow(scored)) scored candidate edges)")

    # ── Step 4: anchor to known dosed compounds, then build the confirmed
    #            network (thresholded edges + BFS generation from anchors) ──
    known_compounds = CSV.read("known_compounds.csv", DataFrame)  # edit this file with YOUR dosed compounds
    anchors = identify_anchor_features(Float64.(df.MeasMass), df.feature_id, known_compounds; ppm_tol=10.0)
    println("  Anchor matches: $(nrow(anchors))")

    if nrow(anchors) == 0
        println("  No known-compound anchors found in this sample — check known_compounds.csv")
        println("  masses/ppm_tol, or this sample may just not contain any of the listed dosed compounds.")
    else
        nodes, net_edges = build_confirmed_network(
            scored, anchors, lookup;
            score_col=:cnl_combined, min_score=0.2,
            n_matched_col=:n_matched_cnl, min_n_matched=1,
        )
        nodes_name = "nodes_" * splitext(fname)[1] * ".csv"
        edges_name = "Edges folder/network_edges_" * splitext(fname)[1] * ".csv"
        #CSV.write(nodes_name, nodes)
        CSV.write(edges_name, net_edges)
        println("  Saved $nodes_name ($(nrow(nodes)) nodes) and $edges_name ($(nrow(net_edges)) confirmed edges)")
        println("  -> import both into Cytoscape (File > Import > Network from File, then Table from File")
        println("     for node attributes) to actually SEE the network, colour nodes by `generation`,")
        println("     and highlight `is_anchor` nodes as your known parent drugs.")
    end
end

println("\nDone. Each edges_<sample>.csv is a network edge list:")
println("  parent_id, metabolite_id            -> the two nodes")
println("  transformation, delta_observed/expected, ppm_error -> why they were paired (MS1 evidence)")
println("  mz_combined, cnl_combined           -> spectral confirmation strength (MS2 evidence)")
println("  n_matched_mz, n_matched_cnl          -> how many peaks/bins each score is actually based on")
println("\nSuggested starting filter for 'confirmed' edges: cnl_combined > 0.5 || mz_combined > 0.5,")
println("plus n_matched_* >= 3 to avoid over-trusting scores computed from 1-2 lucky matches.")