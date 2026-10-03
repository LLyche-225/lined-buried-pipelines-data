"""Five actual mesh sizes against a finer independent beam FE reference."""
from pathlib import Path
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(ROOT/"Step_08A_Draft_Confirmation"))
import run_draft_confirmation as dc


def main():
    if (HERE/"completion.json").exists():
        print("Registered mesh results already complete; no overwrite.")
        return
    sizes=[.4,.2,.1,.05,.025]
    eng=dict(dc.base_config()["engineering"],load_force_N=dc.FORCE_FE)
    dc.write_json(HERE/"registration.json",{
        "registered_before_execution":True,"meshes_m":sizes,"reference_mesh_m":.0125,
        "cases":list(dc.SCENARIOS),"engineering":eng,"score_interval_m":[-12,12],
        "fitted_convergence_order":False,"inverse_runs_changed":False,
        "script_sha256":dc.ni.sha256(Path(__file__))})
    records=[]
    for case in dc.SCENARIOS:
        path=ROOT/"Step_08A_Draft_Confirmation/data"/("FE_"+case+".npz")
        with np.load(path) as f:
            x=f["x"];r=f["retention"];centers=list(f["load_centers_m"])
        roi=abs(x)<=12
        wr,er=dc.fe_response(x,r,centers,eng,h=.0125)
        for h in sizes:
            w,e=dc.fe_response(x,r,centers,eng,h=h)
            records.append({"case_id":case,"mesh_m":h,"reference_mesh_m":.0125,
                "elements":round(100/h),"strain_relative_l2":float(np.linalg.norm(e[roi]-er[roi])/np.linalg.norm(er[roi])),
                "deflection_relative_l2":float(np.linalg.norm(w[roi]-wr[roi])/np.linalg.norm(wr[roi])),
                "reference_input_sha256":dc.ni.sha256(path)})
        print(case+" five meshes completed",flush=True)
    dc.write_csv(HERE/"fe_mesh_refinement.csv",records)
    dc.write_json(HERE/"completion.json",{"status":"COMPLETE","scored_meshes":len(records),
        "reference_solves":3,"total_forward_solves":18,"independent_physical_validation":False,
        "max_h005_strain_error":max(r["strain_relative_l2"] for r in records if r["mesh_m"]==.05)})


if __name__=="__main__":main()
