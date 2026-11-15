import sys, json, urllib.request

ids = sys.argv[1:]
for jid in ids:
    with urllib.request.urlopen(f"http://127.0.0.1:8017/api/jobs/{jid}", timeout=30) as r:
        d = json.load(r)
    j = d["result"]
    mp = {}
    for i in j["preservation"]["items"]:
        mp[i["aspect"].split(" (")[0]] = (i.get("metric") or {})
    print(f"=== {jid}  steps={j['steps']}  guidance={j.get('guidance_scale')} ===")
    c = mp.get("Colour similarity", {})
    print(f"colour  hist_corr={c.get('histogram_correlation')}  deltaE={c.get('dominant_delta_e')}")
    p = mp.get("Pattern / print preservation", {})
    print(f"pattern edge_ratio={p.get('density_ratio')}  orb_good={p.get('good_matches')}  inlier={p.get('inlier_ratio')}")
    cl = mp.get("Overall garment appearance", {})
    print(f"clip    cosine={cl.get('clip_cosine')}")
    oci = mp.get("Logo / text preservation", {})
    print(f"logo    recall={oci.get('recall')}  garment_tokens={oci.get('garment_tokens')}  result_tokens={oci.get('result_tokens')}")
    rel = {i["check"]: i["status"] for i in j["reliability"]["items"]}
    print("reliability:", rel)
