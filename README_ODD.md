## ODD Documentation

The current ODD for this work is under review and will be published soon.

### Resources

ODD Publications

- [ODD Description Paper](https://www.jasss.org/23/2/7/S1-ODD.pdf)
- [Original ODD Paper](https://www.sciencedirect.com/science/article/abs/pii/S0304380006002043?via%3Dihub)
- [ODD First Update, 2010](https://www.sciencedirect.com/science/article/abs/pii/S030438001000414X)
- [ODD Second Update, 2020](https://www.jasss.org/23/2/7.html)

Original RTI ODD

- [2020 COVID-19 Model](https://arxiv.org/abs/2106.04461)


### Recreate Results

To create the output used in the ODD, run the following:

```bash
source python_env/bin/activate
python src/targets.py 
```

This will take 5-6 minutes to run on standard laptop.

Go to `experiments/base/scenario_base_full/run_0` to see the model output. Files are labeled by the pattern they correspond to.