from __future__ import annotations
import json
import pandas as pd
from pathlib import Path
import pprint

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np



class TrackerAna:

    def __init__(
        self,
        csv_path: str | Path,
        baselines_path : str | Path,
        chunksize: int = 900_000
    ):
        self.csv_path = Path(csv_path)
        self.chunksize = chunksize
        self.total_rows: Optional[int] = None
        self.columns: list[str] = []

        self.baselines_path = Path(baselines_path)
        self.pedestals = {}
        self.noise     = {}

        self.vec_baselines = {}

        self.low_threshold = 2.5
        self.high_threshold = 4.0

        self.n_processed_batches = 0

        self.samples = list(range(0,3))
        self.hybrids = list(range(0,4))
        self.channels = list(range(0,640))

        self.hit_channels = {}
        self.hits_per_event = {}
        self.station_1_x = []
        self.station_1_y = []
        self.station_2_x = []
        self.station_2_y = []


        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.init_baselines();

        """
        with open(self.baselines_path,'r') as file:
            _baselines = json.load(file)

            for _hybrid in _baselines:
                if not self.baselines.get(int(_hybrid)):
                    self.baselines[int(_hybrid)] = {}
                    self.noise[int(_hybrid)] = {}
                for _channel in _baselines[_hybrid]:
                    if not self.baselines[int(_hybrid)].get(_channel):
                        self.baselines[int(_hybrid)][_channel] = {}
                        self.noise[int(_hybrid)][_channel] = {}
     
                    avg_baseline = 0.0
                    avg_noise    = 0.0
                    for _sample in _baselines[_hybrid][_channel]:
                        avg_baseline += _baselines[_hybrid][_channel][_sample]['baseline']
                        avg_noise    += _baselines[_hybrid][_channel][_sample]['noise']

                    self.baselines[_hybrid][_channel]['baseline'] = avg_baseline / len(_baselines[_hybrid][_channel])
                    self.baselines[_hybrid][_channel]['noise'] = avg_noise / len(_baselines[_hybrid][_channel])
        """

    def init_baselines(self):

        if not self.baselines_path.exists():
            raise FileNoteFoundError(f"Baselines not found {self.baselines_path}")


        baselines = {}
        noise = {}

        with open(self.baselines_path,'r') as file:
            _baselines = json.load(file)

            for _hybrid in _baselines:
                if not baselines.get(int(_hybrid)):
                    baselines[int(_hybrid)] = {}
                    noise[int(_hybrid)]     = {}
                for _channel in _baselines[_hybrid]:
                    if not baselines[int(_hybrid)].get(_channel):
                        baselines[int(_hybrid)][_channel] = {}
                        noise[int(_hybrid)][_channel]     = {}

                    data = _baselines[_hybrid][_channel]
                    avg_baseline = sum(data[s]['baseline'] for s in data) / len(data)
                    avg_noise    = sum(data[s]['noise']    for s in data) / len(data)

                    baselines[int(_hybrid)][_channel] = avg_baseline
                    noise[int(_hybrid)][_channel] = avg_noise 


        pprint.pprint(baselines)

        self.pedestals = (
            pd.Series(baselines) 
              .apply(pd.Series) 
              .stack()
              .astype(int)
        )
        self.pedestals.index = pd.MultiIndex.from_tuples(
           [(hybrid, int(chan)) for hybrid, chan in self.pedestals.index],
           names=['Hybrid', 'pchannel']
        )

        self.noise = (
            pd.Series(noise)
              .apply(pd.Series)
              .stack()
              .astype(int)
        )
        self.noise.index = pd.MultiIndex.from_tuples(
           [(hybrid, int(chan)) for hybrid, chan in self.noise.index],
           names=['Hybrid', 'pchannel']
        )


    def _get_total_rows(self) -> int:
        """Count total rows once (fast enough for 20–50M rows)"""
        if self.total_rows is None:
            with open(self.csv_path, "rb") as f:
                self.total_rows = sum(1 for _ in f) - 1  # -1 for header
        return self.total_rows

    def find_hits(self,_event,hits):

        event = _event.copy()

        event['pedestal'] = pd.MultiIndex.from_frame(event[['Hybrid', 'pchannel']]).map(self.pedestals)
        event['noise']    = pd.MultiIndex.from_frame(event[['Hybrid', 'pchannel']]).map(self.noise)


        event[['sigma0', 'sigma1', 'sigma2']] = (
                   event[['sample0', 'sample1', 'sample2']].sub(event['pedestal'], axis=0)
                                                           .div(event['noise'], axis=0)     )

        normalized = event[ ['sigma0','sigma1','sigma2'] ]

        # All three > 2σ
        all_above_2sigma = (normalized > self.low_threshold).any(axis=1)

        # At least one > 4σ
        at_least_one_above_4sigma = (normalized > self.high_threshold).any(axis=1)

        # Combined condition

        #event['signal_candidate'] = all_above_2sigma & at_least_one_above_4sigma
        event['signal_candidate'] = all_above_2sigma 
     
        candidates = event[event['signal_candidate']]
        for idx,row in candidates.iterrows():
            hy = row['Hybrid']
            ch = row['pchannel']
            if not hits.get(hy):
                 hits[hy] = {}
            if not hits[hy].get(ch):
                 hits[hy][ch] = []

            data ={}
            data['apv_trigger'] = row['APV_trigger']
            data['sigma0'] = row['sigma0'] 
            data['sigma1'] = row['sigma1']
            data['sigma2'] = row['sigma2']

            hits[hy][ch].append(data)

        """
        #for row in event.to_numpy():
        for idx,row in event.iterrows():

            sigmas = {}

            hy = int(row['Hybrid'])
            #hy = str(row[2])
            ch = str(row['pchannel'])
            #ch = str(int(row[3]))
            trig = str(row['APV_trigger'])
            #trig = str(row[1])

            print("HELLO",self.baselines[hy][ch],row['pedestal'])
            continue

            if not self.baselines[hy].get(ch):
                continue

            if self.baselines[hy][ch]['noise']<50.0 or self.baselines[hy][ch]['noise']>120.0:
                continue

            for sample_idx in self.samples:
                sample = row[4+sample_idx]
                #sample = row[f'sample{sample_idx}']

                #print(self.baselines[hy][ch]['baseline'],row[9])
                #print(sample,trigger,self.baselines[hy][ch]['baseline'],self.baselines[hy][ch]['noise'])
                subtracted_sample = sample - self.baselines[hy][ch]['baseline']
                if subtracted_sample < 0:
                  continue

                n_sigma = subtracted_sample / self.baselines[hy][ch]['noise']

                sigmas[sample_idx] = n_sigma 

            n_low  = 0
            n_high = 0
            for _sigma_idx in sigmas:
              sigma = sigmas[_sigma_idx]
              if sigma>self.low_threshold:
                  n_low+=1
              if sigma>self.high_threshold:
                  n_high+=1

            if n_low>2 and n_high>0:
                 if not hits.get(hy):
                     hits[hy] = {}

                 hits[hy]['n_low'] = n_low
                 hits[hy]['n_high'] = n_high
                 hits[hy]['apv_trigger'] = trig
                 hits[hy]['channel'] = ch
        """
    def iter_batches(self, show_progress: bool = True) -> Iterator[pd.DataFrame]:

        for batch_df in pd.read_csv(self.csv_path,
                                    chunksize=self.chunksize,
                                    usecols=['event_count','head', 'error', 'APV_trigger', 'Hybrid', 'pchannel', 'sample0', 'sample1', 'sample2']):
            self.n_processed_batches += 1
            yield batch_df

    def process_row(self, row: tuple | Dict[str, Any]) -> None:
        pass 


    def compute_intersection(self,hits,module1,module2):

        positions = []

        if hits.get(module1) and hits.get(module2):

            for ch_ax in hits[module1]:
                for ch_ste in hits[module2]:
            
                    ch_a = 640 - ch_ax[0] - 320
                    ch_s = ch_ste[0] - 320

                    b_a = 60.0 * ch_a
                    b_s = 60.0 * ch_s
                    y = (b_a - b_s) / 0.1

                    positions.append( (b_a/1000.0,y/1000.0) ) 

        return positions

    def compute_2D_point(self,hits):

        # Check for pairs 1/2 and 3/4
        # Recompute channel number:
        #   Invert axial side (2/4)
        #   Remap with 320 at zero
        # Axial equation: y = + 60 * n_strip_axial
        # Stereo equation: y = 0.1 * x + 60 * n_strip_stereo

        # x = (ba-bs) / tan(0.1) 

        if hits.get('0') and hits.get('1'):

            if hits['0']['apv_trigger'] == hits['1']['apv_trigger']:
                ch_a = 640 - int(hits['0']['channel']) - 320
                ch_s = int(hits['1']['channel']) - 320
            
                b_a = 60.0 * ch_a
                b_s = 60.0 * ch_s
                print(ch_a,ch_s)
                y = (b_a - b_s) / 0.1

                return (b_a/1000.0,y/1000.0)
         
        return None 


    def flush(self,hits):

        all_hits = {}

        """
        amplitude = []
        samples =[]

        for hybrid in hits:

            for trig in hits[hybrid]:

                ch = hits[hybrid][trig]['channel']
                if ch != 391:
                    continue

                sampleID_0 = 3*trig + 0 
                sampleID_1 = 3*trig + 1
                sampleID_2 = 3*trig + 2

                samples.append(sampleID_0)
                samples.append(sampleID_1)
                samples.append(sampleID_2)

                amplitude.append(hits[hybrid][trig]['sigma0'])
                amplitude.append(hits[hybrid][trig]['sigma1'])
                amplitude.append(hits[hybrid][trig]['sigma2'])

                ch = hits[hybrid][trig]['channel']
        """
        for hybrid in hits:
            if not self.hit_channels.get(hybrid):
                self.hit_channels[hybrid] = []

            if not self.hits_per_event.get(hybrid):
                self.hits_per_event[hybrid] = []

            if not all_hits.get(hybrid):
                all_hits[hybrid]= []

            n_hits_this_event = 0

            for ch in hits[hybrid]:

                n_low = 0
                n_high = 0

                max_sample = -1
                max_id = -1

                for trig in hits[hybrid][ch]:
                    for idx in range(0,3):
                        sample_n = trig[f'sigma{idx}']
                        if sample_n>2.0:
                            n_low+=1
                        if sample_n>4.0:
                            n_high+=1
                            _sample_id = 3*trig['apv_trigger'] + idx
                            if sample_n>max_sample:
                                max_sample = sample_n
                                max_id = _sample_id
                if n_low>6 and n_high>3:
                    self.hit_channels[hybrid].append(ch)
                    all_hits[hybrid].append( (ch,max_id) )
                    n_hits_this_event += 1

            self.hits_per_event[hybrid].append(n_hits_this_event)
                    

        beamspot_1 = self.compute_intersection(all_hits,0,1)
        for beamspot in beamspot_1: 
            self.station_1_x.append(beamspot[0])
            self.station_1_y.append(beamspot[1])

        beamspot_2 = self.compute_intersection(all_hits,3,2)
        for beamspot in beamspot_2:
            self.station_2_x.append(beamspot[0])
            self.station_2_y.append(beamspot[1])

        # All done
        hits.clear()
    

    def get_stats(self,var,_list):

        my_array = np.array(_list)
        mean_val = np.mean(my_array)
        rms_val = np.sqrt(np.mean(my_array**2))

        return f"$\\mu_{{{var}}}$ = {mean_val:.3g} mm $\\sigma_{{{var}}}$ = {rms_val:.3g} mm"

    
    def run(self, row_wise: bool = True) -> None:

        hits  = {}

        current_evt_number = 0

        for batch_df in self.iter_batches():

            # Get unique events
            events = batch_df["event_count"].unique()

            # Iterate over unique events
            # Project out each hybrid
            # Subtract pedestals

            for _event in events:

                print(" >> Processing Event : ",_event)

                if _event>current_evt_number:
                    current_evt_number = _event
                    self.flush(hits)

                # If event number is new, self.flush() --> this should add information to plots.

                event = batch_df[ (batch_df['event_count'] == _event) & (batch_df['head'] == 0) & (batch_df['error']==0) ]
                self.find_hits(event,hits)

            if self.n_processed_batches>20:
                break

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4)) 
        ax1.hist2d(self.station_1_x, self.station_1_y,bins=60, cmap='viridis')
        ax1.text(-4, -80, f'{self.get_stats("x",self.station_1_x)} \n{self.get_stats("y",self.station_1_y)} ', fontsize=12, color='white')
        ax1.set_title("Station 1 (upstream)")
        ax1.set_xlabel("X-position [mm]")
        ax1.set_ylabel("Y-position [mm]")
        ax2.hist2d(self.station_2_x, self.station_2_y,bins=60, cmap='viridis')
        ax2.text(-4, -80, f'{self.get_stats("x",self.station_2_x)} \n{self.get_stats("y",self.station_2_y)} ', fontsize=12, color='white')
        ax2.set_title("Station 2 (downstream)")
        ax2.set_xlabel("X-position [mm]")
        ax2.set_ylabel("Y-position [mm]")


        fig, axes = plt.subplots(2,2, figsize=(10, 4))
        axes[0,0].hist(self.station_1_x,alpha=0.7,bins=80, label='Station 1 (x)', color='blue', edgecolor='black', linewidth=0.5, density=False)
        axes[0,0].legend()
        axes[0,1].hist(self.station_1_y,alpha=0.7,bins=80,label='Station 1 (y)', color='red', edgecolor='black', linewidth=0.5, density=False)
        axes[0,1].legend()
        axes[1,0].hist(self.station_2_x,alpha=0.7,bins=80,label='Station 2 (x)', color='blue', edgecolor='black', linewidth=0.5, density=False)
        axes[1,0].legend()
        axes[1,1].hist(self.station_2_y,alpha=0.7,bins=80, label='Station 2 (y)', color='red', edgecolor='black', linewidth=0.5, density=False)
        axes[1,1].legend()
         

        fig, axes = plt.subplots(2,2, figsize=(10, 4))
        axes[0,0].hist(self.hit_channels[0], bins=640,range=(0,640), alpha=0.7, label='Module 1', color='blue', edgecolor='black', linewidth=0.5, density=False)
        axes[0,0].legend()
        axes[0,1].hist(self.hit_channels[1], bins=640,range=(0,640),alpha=0.7, label='Module 2', color='red', edgecolor='black', linewidth=0.5, density=False)
        axes[0,1].legend()
        axes[1,0].hist(self.hit_channels[2], bins=640,range=(0,640),alpha=0.2, label='Module 3', color='green', edgecolor='black', linewidth=0.5, density=False)
        axes[1,0].legend()
        axes[1,1].hist(self.hit_channels[3], bins=640,range=(0,640),alpha=0.2, label='Module 4', color='purple', edgecolor='black', linewidth=0.5, density=False)
        axes[1,1].legend()

        colors_per_hybrid = ['blue', 'red', 'green', 'purple']
        labels_per_hybrid = ['Module 1 (axial)', 'Module 2 (stereo)', 'Module 3 (stereo)', 'Module 4 (axial)']
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        fig.suptitle("Hits per Event per Layer")
        for hybrid, (ax, color, label) in enumerate(zip(axes.flat, colors_per_hybrid, labels_per_hybrid)):
            if self.hits_per_event.get(hybrid):
                max_hits = max(self.hits_per_event[hybrid])
                ax.hist(self.hits_per_event[hybrid], bins=range(0, max_hits + 2),
                        alpha=0.7, label=label, color=color, edgecolor='black', linewidth=0.5)
                ax.set_xlabel("Hits per event")
                ax.set_ylabel("Events")
                ax.legend()
        fig.tight_layout()

        plt.show()

