
export COMET_API_KEY="x49hnrC7hy5M0xxzTUQBAlVrM"

python3 train.py -cn=baseline.yaml \
 dataloader.batch_size=2 \
 trainer.override=True \
 trainer.n_epochs=2 \
 datasets=onebatchtest
