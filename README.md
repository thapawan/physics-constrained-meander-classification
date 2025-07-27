A Physics-Constrained Deep Learning Framework for Meander Classification Using Multi-Sensor Earth Observation Data
Abstract
This study introduces an integrated framework that combines cellular automata (CA) rules, physical principles, and Long Short-Term Memory (LSTM) networks to classify river meanders across diverse spatial and temporal scales. Addressing the limitations of purely data-driven approaches in geomorphology, our methodology ensures predictions adhere to known fluvial principles. We developed a harmonized 10m resolution dataset (2015-2024) by integrating Sentinel-2-derived Modified Normalized Difference Water Index (MNDWI) centerlines, USGS 3DEP topography , CHIRPS  and ERA5 climate data , NLCD land cover , and SMAP-enhanced soil erodibility metrics. The framework incorporates three key innovations: (1) physics-based normalization constraining hydraulic geometry relationships, (2) CA transition rules grounded in established meander evolution theories, and (3) a hybrid LSTM architecture that respects first principles while learning complex temporal patterns. Applied to three Alabama rivers and scaled to 13 southeastern U.S. states, the model achieved 82% accuracy in classifying five meander types (Incipient to Complex) through explicit incorporation of curvature-discharge-slope interactions and human disturbance factors. Validation demonstrated strong agreement (R² = 0.76) between predicted migration potential and field-observed bank erosion rates. This workflow provides a replicable template for coupling process-based knowledge with machine learning in geomorphological applications, offering a pathway for explainable AI in environmental science.   

1. Introduction and Background
River meandering is a fundamental geomorphic process that profoundly shapes floodplain ecosystems, dictates sediment transport pathways, and presents significant risks to human infrastructure [Güneralp et al., 2012]. Accurate meander classification is essential for understanding river behavior, guiding ecological restoration, and informing flood risk management. Recent advances in Earth observation data, particularly from missions like Sentinel-2, providing 10-m multi-spectral bands, and the Surface Water and Ocean Topography (SWOT) mission, offer unprecedented spatial and temporal coverage. Concurrently, deep learning, especially Convolutional Neural Networks (CNNs), has revolutionized image classification [LeCun et et al., 2015] and shown strong performance in hydrological applications [Xu et al., 2022].

Despite these advancements, traditional meander classification methods are time-consuming, subjective, and lack scalability [Constantine et al., 2014]. Purely data-driven machine learning approaches, while powerful, risk violating known fluvial principles [Marcus & Fonstad, 2023] and often suffer from low physical interpretability and limited generalization [Gonzales-Inca et al., 2022]. This study addresses this critical gap by developing a physics-constrained deep learning framework that harmonizes empirical theory with data-driven pattern detection, ensuring predictions are both measurable and theoretically grounded.

2. Research Objectives
This study aims to automate the classification of river meanders into established geomorphic types through a novel deep learning framework. Our approach is distinguished by three key innovations:

Physics-based normalization constraining hydraulic geometry relationships (e.g., discharge ≈ width × velocity × slope; Leopold & Maddock, 1953).

Cellular Automata (CA) transition rules rigorously grounded in established meander evolution theories, specifically Brice’s (1975) meander evolution stages and Nanson & Hickin’s (1986) cutoff criteria.   

A hybrid Long Short-Term Memory (LSTM) architecture that embeds hydraulic equations directly into the neural network, respecting first principles while learning complex temporal patterns.   

3. Methodology Overview
The framework involves constructing a harmonized 10m resolution multi-sensor Earth observation dataset (2015-2024), applying physics-based normalization to features, and then utilizing a hybrid LSTM-CA model for meander classification. The model's performance is validated through classification accuracy against field-verified data and correlation with observed bank erosion rates.

4. Detailed Data Sources and Characteristics
This project leverages a comprehensive suite of multi-sensor Earth observation datasets, harmonized to a 10m spatial resolution for the period 2015-2024. Data acquisition and preprocessing were performed using Google Earth Engine.

Data Product	Source/Sensor	Original Spatial Resolution	Temporal Resolution/Coverage	Key Variables Used	Purpose in Study
Sentinel-2 MSI	COPERNICUS/S2_SR	10m (B2, B3, B4, B8), 20m (B5, B6, B7, B8a, B11, B12), 60m (B1, B9, B10)	5 days combined constellation revisit (annual composites 2015-2024)	MNDWI (B3, B11), NDVI (B8, B4)	River centerlines, water extent, vegetation trend
USGS 3DEP DEM	USGS/3DEP/10m	1/3 arc-second (~10m)	Static (survey dates 1923-2017)	Slope	Topographic influence on meander dynamics
CHIRPS Climate Data	UCSB-CHG/CHIRPS/DAILY	0.05° (~5km)	Daily (annual sum 2015-2024)	Precipitation	Climate influence, discharge proxy
ERA5-Land Climate Data	ECMWF/ERA5_LAND/DAILY	~9km grid spacing	Daily (annual mean 2015-2024)	Temperature	Climate influence
NLCD Land Cover	USGS/NLCD	30m	
Inconsistent (every 2-3 years), 2001-2021    

Developed, cultivated, forest	Human impacts, riparian influence
SSURGO Soil Data	USDA/NRCS/SSURGO/v2	1:12,000; 1:63,360	Static (e.g., 2016)	K-factor	Soil erodibility
SMAP Soil Moisture	NASA/SMAP/SPL4SMGP/007	9km Equal-Area Scalable Earth Grid	3-hourly time-averaged, 2-3 days global coverage (annual mean 2015-2024)	Surface soil moisture ('sm_surface')	Enhance soil erodibility
GDAT Dams	projects/sat-io/open-datasets/GDAT/GDAT_V1_DAMS	Point data	
GRanD v1.1/1.3 (most added 2000-2016)    

Dam location (for distance)	Human impacts (flow alteration)
5. Computational Modeling Approach
The core of this framework is a hybrid LSTM-Cellular Automata (CA) model.

Cellular Automata (CA): CA are discrete models that generate large-scale patterns from small-scale local processes, operating on a grid structure. They are used here to enforce physically plausible state changes (e.g., straight → incipient → mature) and act as a "sanity check," overriding AI outputs when they deviate from empirical thresholds (e.g., preventing a low-curvature reach from being classified as "Complex").   

Long Short-Term Memory (LSTM) Networks: LSTMs are a type of recurrent neural network highly effective for sequence generation and time series prediction. In this framework, the LSTM component learns complex, non-linear temporal patterns in meander evolution, while embedding hydraulic equations directly into its structure to ensure predictions respect mass and energy conservation. LSTMs are well-suited for hydrological applications like rainfall-runoff modeling and flood forecasting using discharge and rainfall data.   

Physics-Informed Deep Learning: This approach combines the strengths of data-driven methods with the constraints of physical laws. Our model embeds hydraulic equations and geomorphic principles directly into the learning process, ensuring predictions are both accurate and physically consistent.   

6. Expected Outcomes and Broader Significance
This framework achieved 82% accuracy in classifying five meander types and demonstrated strong agreement (R² = 0.76) between predicted migration potential and field-observed bank erosion rates. Its successful application across 13 southeastern U.S. states highlights its significant generalization capabilities and scalability. This work provides a replicable template for "explainable AI" in geomorphology, balancing theoretical fidelity with predictive power. The resulting classifications and predictive capabilities can directly inform hydraulic models, improve flood prediction accuracy, and support resilient land use planning in dynamic riverine environments.

7. Repository Structure and Usage Guidelines
This repository is organized to facilitate reproducibility and understanding of the physics-constrained deep learning framework.
.
├── data/
│   ├── raw/              # Raw input data (e.g., downloaded satellite imagery, DEMs)
│   └── processed/        # Intermediate processed data (e.g., MNDWI centerlines, normalized features)
├── src/
│   ├── models/           # Python scripts for model definition (e.g., LSTM-CA architecture)
│   ├── utils/            # Utility functions (e.g., data normalization, CA rules)
│   └── main.py           # Main script for data extraction, processing, training, and evaluation
├── notebooks/            # Jupyter notebooks for exploratory data analysis or result visualization
├── results/              # Output results (e.g., classification maps, performance metrics, plots)
├── config/               # Configuration files (e.g., model hyperparameters, study area definitions)
├──.gitignore            # Specifies files/directories to ignore in Git
├── README.md             # This file
└── requirements.txt      # Python dependencies


**To use this framework:**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-org/model-geomorph-meander-classification-physics-dl.git](https://github.com/your-org/model-geomorph-meander-classification-physics-dl.git)
    cd model-geomorph-meander-classification-physics-dl
    ```
2.  **Set up Python environment:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows:.\venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  **Google Earth Engine Setup:**
    Ensure you have authenticated your Google Earth Engine account. Follow the official GEE Python API setup guide if you haven't already.
4.  **Data Preparation:**
    The `main.py` script (or specific data download scripts in `src/utils/`) will handle the data extraction and preprocessing from Google Earth Engine. Ensure your GEE account has access to the specified datasets.
5.  **Run the main script:**
    ```bash
    python src/main.py --config config/default_config.yaml
    ```
    Adjust parameters in `config/default_config.yaml` as needed for different experiments or study areas.

### 8. Contribution and Citation

Contributions to this project are welcome. Please follow standard GitHub pull request workflows. For major changes, open an issue first to discuss what you would like to change.

If you use this framework or its components in your research, please cite the associated publication (details to be added upon publication).

### 9. License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## Research Script (`src/main.py` - based on provided snippet)

The following Python script snippet illustrates the core data extraction and physics-based normalization components of the framework. The full `main.py` script would integrate these functions with the LSTM-CA model definition, training, and evaluation.



