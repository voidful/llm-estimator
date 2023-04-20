import React, {useState} from 'react';
import {Box, Button, Container, Grid, TextField, Typography} from '@mui/material';
import {ModelEstimator} from './ModelEstimator';

function App() {
    const [modelId, setModelId] = useState('');
    const [modelParams, setModelParams] = useState({
        nh: 0, dh: 0, de: 0, dff: 0, N: 0,
    });
    const [trainingParams, setTrainingParams] = useState({
        L: 512, S: 1000, B: 10,
    });
    const [gpuParams, setGpuParams] = useState({
        tflopsPerGpu: 30, numGpus: 2, gpuHourlyCost: 10
    });

    const parameterDescriptions = {
        nh: 'Number of Heads',
        dh: 'Dimension of Head',
        de: 'Dimension of Embedding',
        dff: 'Dimension of Feed Forward Network',
        N: 'Number of Layers',
        L: 'Average Input Sequence Length',
        S: 'Training Steps',
        B: 'Batch Size',
        tflopsPerGpu: 'T FLOPS per GPU',
        numGpus: 'Number of GPUs',
        gpuHourlyCost: 'GPU Hourly Cost',
    };

    const mapConfigToParams = (config) => {
        return {
            nh: config.n_head,
            dh: config.hidden_size / config.n_head,
            de: config.hidden_size,
            dff: config.n_inner || config.hidden_size * 4, // Use n_inner if provided, else use hidden_size * 4 as a default
            N: config.n_layer,
        };
    };

    const handleModelIdChange = (event) => {
        setModelId(event.target.value);
    };

    const fetchModelInfo = async () => {
        const response = await fetch(`https://huggingface.co/${modelId}/raw/main/config.json`);
        const config = await response.json();
        const newParams = mapConfigToParams(config);
        setModelParams(modelParams => ({...modelParams, ...newParams}));
    };

    const handleModelInputChange = (event) => {
        const {name, value} = event.target;
        setModelParams({
            ...modelParams, [name]: parseInt(value, 10),
        });
    };

    const handleTrainingInputChange = (event) => {
        const {name, value} = event.target;
        setTrainingParams({
            ...trainingParams, [name]: parseInt(value, 10),
        });
    };

    const handleGpuInputChange = (event) => {
        const {name, value} = event.target;
        setGpuParams({
            ...gpuParams, [name]: parseInt(value, 10),
        });
    };

    return (<Container maxWidth="lg">
        <Box my={4}>
            <Typography variant="h4" component="h1" gutterBottom>
                LLM Estimator
            </Typography>
            <Grid container spacing={2}>
                <Grid item xs={12}>
                    <Typography variant="h4" component="h2" gutterBottom>
                        Model Parameters
                    </Typography>
                </Grid>
                <Grid item xs={12} md={4}>
                    <TextField
                        label="Model ID"
                        value={modelId}
                        onChange={handleModelIdChange}
                        fullWidth
                        variant="outlined"
                    />
                </Grid>
                <Grid item xs={12}>
                    <Button variant="contained" color="primary" onClick={fetchModelInfo}>
                        Fetch Model Info </Button> </Grid>
                {Object.entries(modelParams).map(([name, value]) => (<Grid item xs={12} md={4} key={name}>
                    <TextField
                        type="number"
                        label={`${parameterDescriptions[name]} (${name.toUpperCase()})`}
                        name={name}
                        value={value}
                        onChange={handleModelInputChange}
                        fullWidth
                        variant="outlined"
                    /> </Grid>))}
                <Grid item xs={12}>
                    <Typography variant="h4" component="h2" gutterBottom>
                        Training Parameters
                    </Typography>
                </Grid>
                {/* Training parameter inputs */}
                {Object.entries(trainingParams).map(([name, value]) => (<Grid item xs={12} md={4} key={name}>
                    <TextField
                        type="number"
                        label={`${parameterDescriptions[name]} (${name.toUpperCase()})`}
                        name={name}
                        value={value}
                        onChange={handleTrainingInputChange}
                        fullWidth
                        variant="outlined"
                    /> </Grid>))}
                <Grid item xs={12}>
                    <Typography variant="h4" component="h2" gutterBottom>
                        GPU Parameters
                    </Typography>
                </Grid>
                {Object.entries(gpuParams).map(([name, value]) => (<Grid item xs={12} md={4} key={name}>
                    <TextField
                        type="number"
                        label={`${parameterDescriptions[name]} (${name.toUpperCase()})`}
                        name={name}
                        value={value}
                        onChange={handleGpuInputChange}
                        fullWidth
                        variant="outlined"
                    /> </Grid>))}
                <Grid item xs={12}>
                    <ModelEstimator params={{...modelParams, ...trainingParams, ...gpuParams}}/> </Grid>
            </Grid>
        </Box>
        <h2>Understanding the Trade-offs and Recommendations</h2> <p>
        When training a model, there is a trade-off between training time, loss, and the number of dataset
        tokens. Increasing the number of dataset tokens may lead to a lower loss, but at the cost of increased
        training time. Conversely, reducing the number of dataset tokens may lead to faster training but higher
        loss. </p> <p>
        To strike a balance between these factors, consider the following recommendations: </p>
        <ol>
            <li>Choose an appropriate model size based on your computational resources and training time
                constraints.
            </li>
            <li>Optimize hyperparameters such as learning rate, batch size, and optimizer to improve training
                efficiency.
            </li>
            <li>Regularly monitor the training loss and validation loss to avoid overfitting and underfitting.</li>
            <li>Utilize techniques such as data augmentation, transfer learning, and early stopping to improve model
                performance and reduce training time.
            </li>
            <li>Perform cost-benefit analysis to determine the optimal training duration and dataset size that align
                with your project's budget and goals.
            </li>
        </ol>
        <p>
            By carefully considering these recommendations, you can make well-informed decisions about model
            training and achieve a better balance between training time, loss, and dataset tokens. </p>
    </Container>);
}

export default App;