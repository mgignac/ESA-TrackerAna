

from TrackerAna import TrackerAna

# ————— Example usage —————
if __name__ == "__main__":

    ana = TrackerAna("/u1/mgignac/sw/dev/tracker-qa/build/Run111.csv",
#                     "/u1/mgignac/sw/tracker-qa/scripts/baselines.json") # Old baselines
                      "/u1/mgignac/sw/tracker-qa/scripts/baselines_Dec11_Vbias65V.json") 
    ana.run()
