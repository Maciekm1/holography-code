# holography-code
PhD code for Holographic Microscopy and 3D tracking.

After cloning the respository, you can install the necessary python libraries using the holography_pks file. Please run:

```
bash holography_pks
```
User imput is needed to compelted the installation.

# Main scripts to perform holgoraphy
## Rayleigh-Sommerfeld and Modified Propagation
**positions_batch_multiprocess.py**
This script calculated the positions of cells from a holographic video recording.
When the script is run, and interactive window is open to allow the user to select propagation method, and corresponding parameters. Export to CSV option is also included. You can find an alternative Juypter notebook (**positions_batch_multiprocess.ipynb**) for a more interactive experience.

### Gradient stack tool check
**gradient_stack_check.py**
This tool allows the user to interactively find the best threshold to filter propagation stack after applying Sobel-type filter. This should be run before running **positions_batch_multiprocess.py**. An alternative Jupyter notebook (**gradient_stack_check.ipynb**) is also present.

### Gradient stack tool check
**gradient_stack_check_mod_prop.ipynb**
This tool allows the user to interactively find the best threshold to filter propagation stack after applying Sobel-type filter when using the modified propagator. This should be run before running **positions_batch_multiprocess.py**.

## 3D Tracking
**make_tracks.py**
This script should be run after obaining the results from **positions_batch_multiprocess.py**. An alternative Jupyter notebook (**make_tracks.ipynb**) is also present.

## Helper Functions
All needed helper functions are contained in **functions.py**.


