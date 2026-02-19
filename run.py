

from TrackerAna import TrackerAna

# ————— Example usage —————
if __name__ == "__main__":

    ana = TrackerAna("/sdf/data/hps/users/mgignac/hardware/data/LDMX/Run111.csv",
                      "/sdf/data/hps/users/mgignac/hardware/data/LDMX/baselines_Dec11_Vbias65V.json")
    ana.run()
