# Main function
Test_Ea_Alg.m:
# Algorithm framwork within Test_Ea_Alg.m
G_overlay = gen_overlay_netw(n_V, N_links, G_type, modelType, F_calculated): generate G_overlay as the meta data structure for overlay network
biLevel_upper(G_overlay, M1, M2, "boyd_greedy"): upper-level function for all algorithms. The 4th argument is the algorithms' name in string. This function invoke lower-level algorithm.
----SDR_lambda2_equalWeight: minimize \lambda_2(L) with equal weight.
----SDR_rho_equalWeight: SDR of (23)
----SCA_both_rho_ew: SCA-23
----greedy_boyd_lower: Boyd's greedy
# G_overlay = gen_overlay_netw()
eo: overlay link
eu: underlay link
vu: underlay nodes indices
vo: overlay nodes indices
pair: undirected pair <s,t> in Ea
link: directed overlay link. #link = 2*#pair
st2xx: a Hashmap (containers.Map in Matlab) for src-dst pair <s,t> to the associated xx, i.e., pair or link. For example, G_overlay.st2spr("5 7") returns an array indicating the routing path from 5->7. I use 'strjoin(string([u, v]))' to generate the string key "u v".
TensorFlow 2.x
NumPy
# gen_tau_from_EaRecord: Main function for tau generation, especially the one w/ overlay routing.
Will call function min_tau_w_overlayRouting(), which is the function solving (5).

