#!/bin/bash
#SBATCH --job-name=267-SP-21-232-C2_realignment_v2    # Job name
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jiasong@coh.org     # Where to send mail
#SBATCH -n 16                          # Number of cores
#SBATCH -N 1-1                        # Min - Max Nodes
#SBATCH -p bigmem                        # default queue is all if you don't specify
#SBATCH --mem=512G                      # Amount of memory in GB
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log   # Standard output and error log


## VARIABLES ##
human_transcriptome="/coh_labs/dits/Resources/10x/reference/refdata-gex-GRCh38-2020-A/"
probe_set="/coh_labs/dits/Resources/10x/probes/human_transcript_v2/Visium_Human_Transcriptome_Probe_Set_v2.0_GRCh38-2020-A.csv"
project_path="/coh_labs/dits/nsong/Craig_VisiumHD_20241218"

gbm_fastqs="/coh_labs/dits/nsong/Craig_VisiumHD_20241218/fastqs/267-SP-21-232-C2"
loupe_alignment="/coh_labs/dits/nsong/Craig_VisiumHD_20241218/run1/267-SP-21-232-C2/outs/H1-MY97TWM-A1-fiducials-image-registration.json"
slide_image="/coh_labs/dits/nsong/Craig_VisiumHD_20241218/visium_ndpi/267_SP_21_232_C2-003_lzw.tiff"
cytaimage="/coh_labs/dits/nsong/Craig_VisiumHD_20241218/images/images/267-SP-21-232-C2_H1-MY97TWM_A1.tif"

## COMMANDS ##
cd ${project_path}
spaceranger count --id="267-SP-21-232-C2_realignment_v2" \
                  --description="rerunning" \
                  --transcriptome=${human_transcriptome} \
                  --probe-set=${probe_set} \
                  --fastqs=${gbm_fastqs} \
                  --cytaimage=${cytaimage} \
                  --image=${slide_image} \
                  --loupe-alignment=${loupe_alignment} \
                  --localcores=16 \
                  --localmem=128 \
                  --create-bam=true \
                  --area=A1 \
                  --slide=H1-MY97TWM
 