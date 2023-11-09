# Indizio 
Visualization dashboard for presence/absence data, distance matrices, and phylogenetic trees.

## Installation
Installation not currently supported for Windows.

Download the repository
```
git clone https://github.com/beiko-lab/indizio.git
```

Suggested: Create a conda environment to manage dependancies. This requires Anaconda or Miniconda.

### Linux and MacOS
The developers intend to create a bioconda recipe at a later date.
For now, to install:
```
conda create -n indizio pandas networkx tqdm
conda activate indizio
conda install -c anaconda pillow
conda install -c conda-forge dash dash-bootstrap-components dash_cytoscape
```


## Usage
The major feature of the Inidizio tool is the interactive Dash application.
The Indizio dash tool is primarily used to identify and visualize correlations among features in a sample set.

### Set-up script
Indizio is flexible with the number of files that can be used as input. As a bare minimum, Indizio requires either a presence/absence table of features in samples or a feature-wise distance matrix. If a presence/absence table is supplied, Indizio will calculate a simple Pearson correlation among features.

Users may supply as many distance matrices as the would like. During the set-up script, they will be asked to name each distance matrix.

Users may also supply metadata. These metadata are meant to be correlations of features to specific labels. At this time, only feature-wise metadata are supported.

Finally, users may upload a phylogenetic tree or similar sample dendrogram file. If both a tree and sample-wise feature presence/absence table are uploaded, Indizio will generate clustergrams.

To run the set-up script, simply activate your conda environment and invoke the script. This script will create a file which should be provided to the Indizio Dash application as input:
```
conda activate indizio
python3 make_input_sheet.py
```


### Dash Application
The Indizio tool contains a simple to use set-up script that will ask you a series of prompts and  will subsequently generate the input file for the Dash application (see above). The final step of the set-up script will have asked you to name your input file.

Once the input file is generated, launch the Indizio Dash application:

```
conda activate indizio #if you have not done so already
python3 app.py myInputSheet.csv
```
Next, launch your preferred web browser and navigate to http://localhost:8050/ .
