import {all, create} from "mathjs";

// Function to calculate FLOPS (Floating Point Operations Per Second) based on input parameters
// Returns FLOPS in teraflops (TFLOPS)
export const calculateFlops = (params) => {
    const {
        S: trainingSteps,
        B: batchSize,
        N: numLayers,
        nh: numHeads,
        dh: headDim,
        de: embeddingDim,
        L: inputLength,
        dff: feedForwardDim,
    } = params;

    // Calculate FLOPS per token per layer for Multi-Head Attention (MHA), Softmax and Element-wise operations, and Feed-Forward Network (FFN)
    const MHA = (nh, dh, de, L) => 2 * nh * (dh ** 2) * (3 + 2 * L);
    const calculateSoftmaxAndElementWise = (de) => 2 * (de ** 2);
    const FFN = (de, dff) => 16 * (de ** 2);
    const flopsPerTokenPerLayer = MHA(numHeads, headDim, embeddingDim, inputLength) + calculateSoftmaxAndElementWise(embeddingDim) + FFN(embeddingDim, feedForwardDim);

    // Calculate total FLOPS based on input parameters and FLOPS per token per layer
    const flops = trainingSteps * batchSize * inputLength * numLayers * flopsPerTokenPerLayer;

    // Calculate memory cost per token per layer using a memory latency factor obtained from a research paper
    const memoryLatencyFactor = 0.05375;
    const memoryCostPerTokenPerLayer = memoryLatencyFactor * flopsPerTokenPerLayer;

    // Calculate total memory cost based on input parameters and memory cost per token per layer
    const memoryCost = trainingSteps * batchSize * inputLength * numLayers * memoryCostPerTokenPerLayer;

    // Calculate total FLOPS and memory cost and return FLOPS in teraflops (TFLOPS)
    const totalFlopsAndMemoryCost = flops + memoryCost;
    return totalFlopsAndMemoryCost / (10 ** 12);
};

// Function to calculate training time based on FLOPS and input parameters
// Returns training time in hours
export const calculateTrainingTime = (flops, params) => {
    const {tflopsPerGpu, numGpus} = params;

    // Calculate training time in seconds
    const trainingTime = flops / (tflopsPerGpu * numGpus);

    // Convert training time to hours and return
    return trainingTime / 3600;
};

// Function to calculate training cost based on training time and input parameters
// Returns training cost in USD
export const calculateTrainingCost = (trainingTime, params) => {
    const {gpuHourlyCost} = params;

    // Calculate training cost based on training time and GPU hourly cost
    return trainingTime * gpuHourlyCost;
};

// Create a mathjs instance with all functions loaded
const math = create(all);

// Parameters for Chinchilla scaling laws
const A = 406.4;
const B = 410.7;
const E = 1.69;
const alpha = 0.32;
const beta = 0.28;

// Function to calculate loss based on tokens seen using Chinchilla scaling laws
const lossFunction = (tokensSeen) => {
    // Calculate optimal values for N and D based on tokens seen and parameters
    const N_opt = math.pow((alpha * A) / (beta * B), 1 / (alpha + beta)) * math.pow(tokensSeen / 6, beta / (alpha + beta));
    const D_opt = math.pow((tokensSeen / 6) / ((alpha * A) / (beta * B)), alpha / (alpha + beta));

    // Calculate loss based on optimal N and D values and parameters
    return E + A / math.pow(N_opt, alpha) + B / math.pow(D_opt, beta);
};

// Function to calculate range of training steps, tokens seen values, and losses based on input parameters
// Returns object containing arrays of S values and corresponding losses
export const calculateLossData = (params) => {
    // Generate range of S values and calculate corresponding tokens seen values and losses
    const S_values = math.range(1, 1000000, 1000).toArray();
    const tokens_seen_values = math.multiply(S_values, params.B, params.L);
    const losses = tokens_seen_values.map(lossFunction);

    // Return object containing arrays of S values and corresponding losses
    return {
        S_values,
        losses,
    };
};