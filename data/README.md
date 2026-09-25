# Challenge dataset packages

The raw TSVs are **not** committed to git (too large; see `.gitignore`).

They are published as GitHub Release assets:

**https://github.com/sshivanshg/Mlchallenge/releases/tag/dataset-v1**

| Asset | Contents |
| --- | --- |
| `dataset_train.zip` | `dataset/train/train_source{1,2,3}.tsv`, `train_ground_truth.tsv` |
| `dataset_test.zip` | `dataset/test/test_source{1,2,3}.tsv` |

## Extract into this repo

From the repository root:

```bash
bash scripts/download_dataset.sh
```

This downloads both zips and extracts them under `student_resource/dataset/`.

Manual alternative:

```bash
gh release download dataset-v1 -R sshivanshg/Mlchallenge -D /tmp/mlc-data
unzip -o /tmp/mlc-data/dataset_train.zip -d student_resource
unzip -o /tmp/mlc-data/dataset_test.zip -d student_resource
```

Original Drive folder (backup source):  
https://drive.google.com/drive/folders/12U2iIr3vfUHLNrs_Gsvlmi-_Nurgpt2R
