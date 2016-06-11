var Neuraljs = {};

(function (globalAccess) {
    "use strict";

    // Utilities
    function assert(condition, message) {
        if (!condition) {
            message = message || 'Assertion failed';
            if (typeof Error !== 'undefined') {
                throw new Error(message);
            }
            throw message; // Fallback
        }
    }

    var gaussReturn = false,
        gaussValueV = 0.0;

    function gaussRandom() {
        if (gaussReturn) {
            gaussReturn = false;
            return gaussValueV;
        }
        var u, v, r, c;
        u = 2 * Math.random() - 1;
        v = 2 * Math.random() - 1;

        r = (u * u) + (v * v);

        if (r === 0 || r > 1) {
            return gaussRandom();
        }

        c = Math.sqrt(-2 * Math.log(r) / r);
        gaussValueV = v * c; // cached for return on next call
        gaussReturn = true;
        return u * c;
    }
    // returns an array of the size n that is filled with zeros
    function zeros(n) {
        var i;
        if (typeof n === 'undefined' || isNaN(n)) { return []; }
        if (typeof ArrayBuffer === 'undefined') {
            // no browser support
            var arr = [];
            for (i = 0; i < n; i++) { arr[i] = 0; }
            return arr;
        }
        return new Float64Array(n);
    }

    /*
     *  a Matrix object that contains two arrays of weights (which are a simple representation of a mathematical matrix)
     *  w contains the normal weights
     *  dw contains temporal values for the backpropagation process
     */

    var Matrix = function (rows, columns) {
        if (typeof rows === 'undefined' || typeof columns === 'undefined') {
            throw new Error('Undefined not allowed in Matrix initialization')
        }
        this.rows = rows;
        this.columns = columns;
        this.w = zeros(rows * columns);
        this.dw = zeros(rows * columns);
    };
    Matrix.prototype = {
        get: function (row, column) {
            var index = (this.columns * row) + column;
            assert((index >= 0) && (index < this.w.length));
            return this.w[index];
        },
        set: function (row, column, value) {
            var index = (this.columns * row) + column;
            assert((index >= 0) && (index < this.w.length));
            this.w[index] = value;
        },
        toJSON: function () {
            var json = {};
            json.rows = this.rows;
            json.columns = this.columns;
            json.w = this.w;
            return json;
        },
        fromJSON: function (json) {
            var i, n;
            this.rows = json.rows;
            this.columns = json.columns;
            this.w = zeros(this.rows * this.columns);
            this.dw = zeros(this.rows * this.columns);
            n = this.rows * this.columns;
            for (i = 0; i < n; i++) {
                this.w[i] = json.w[i];
            }
        }
    };

    function createMatrixFilledWithOnes(rows, columns) {
        var matrix = new Matrix(rows, columns),
            i;
        for (i = 0; i < matrix.w.length; i++) {
            matrix.w[i] = 1;
        }
        return matrix;
    }

    function createRandomizedMatrix(rows, columns, std) {

        function fillRandom(matrix, low, high) {
            var i, n;
            n = matrix.rows * matrix.columns;
            for (i = 0; i < n; i++) {
                matrix.w[i] = Math.random() * (high - low) + low;
            }
        }

        var matrix = new Matrix(rows, columns);
        fillRandom(matrix, -std, std);
        return matrix;

    }

    function createNegatedCloneMatrix(matrix) {
        var negativeClone = new Matrix(matrix.rows, matrix.columns),
            i;
        for (i = 0; i < matrix.w.length; i++) {
            negativeClone.w[i] = -matrix.w[i];
        }
        return negativeClone;
    }

    /*
     *   a graph is responsible for mathematical operations on matrices. It automatically saves the
     *   necessary backpropagation functions, which will update the dw array of the matrix
     */
    var Graph = function (needsBackpropagation) {
        if (typeof needsBackpropagation === 'undefined') {
            needsBackpropagation = true;
        }
        this.needsBackpropagation = needsBackpropagation;

        // this stores a list of functions that perform the backpropagation in the right order
        this.backprop = [];
    };
    Graph.prototype = {
        performBackpropagation: function () {
            while (this.backprop.length > 0) {
                this.backprop.pop()();
            }
        },
        pluckRow: function (matrix, rowIndex) {
            // returns the row as a column vector
            assert((rowIndex >= 0) && (rowIndex < matrix.rows));
            var columns = matrix.columns,
                out = new Matrix(columns, 1),
                i;
            for (i = 0; i < columns; i++) {
                out.w[i] = matrix.w[columns * rowIndex + i];
            }

            if (this.needsBackpropagation) {
                var backward = function () {
                    for (i = 0; i < columns; i++) {
                        matrix.dw[columns * rowIndex + i] += out.dw[i];
                    }
                    this.backprop.push(backward);
                };
            }
            return out;
        },
        tanh: function (matrix) {
            var out = new Matrix(matrix.rows, matrix.columns),
                n = matrix.w.length,
                i;
            for (i = 0; i < n; i++) {
                out.w[i] = Math.tanh(matrix.w[i]);
            }

            if (this.needsBackpropagation) {
                var backward = function () {
                    var matrixWi;
                    for (i = 0; i < n; i++) {
                        matrixWi = out.w[i];
                        matrix.dw[i] += (1.0 - matrixWi * matrixWi) * out.dw[i];
                    }
                };
                this.backprop.push(backward);
            }
            return out;
        },
        sigmoid: function (matrix) {
            var out = new Matrix(matrix.rows, matrix.columns),
                n = matrix.w.length,
                i;
            for (i = 0; i < n; i++) {
                out.w[i] = 1.0 / (1 + Math.exp(-matrix.w[i]));
            }

            if (this.needsBackpropagation) {
                var backward = function () {
                    var matrixWi;
                    for (i = 0; i < n; i++) {
                        matrixWi = out.w[i];
                        matrix.dw[i] += matrixWi * (1.0 - matrixWi) * out.dw[i];
                    }
                };
                this.backprop.push(backward);
            }
            return out;
        },
        relu: function (matrix) {
            var out = new Matrix(matrix.rows, matrix.columns),
                n = matrix.w.length,
                i;
            for (i = 0; i < n; i++) {
                out.w[i] = Math.max(0, matrix.w[i]);
            }

            if (this.needsBackpropagation) {
                var backward = function () {
                    for (i = 0; i < n; i++) {
                        matrix.dw[i] += matrix.w[i] > 0 ? out.dw[i] : 0.0;
                    }
                };
                this.backprop.push(backward);
            }
            return out;
        },
        mul: function (matrix1, matrix2) {
            assert(matrix1.columns === matrix2.rows, 'matrix dimensions not compatible for multiplication');

            var rows = matrix1.rows,
                columns = matrix2.columns,
                out = new Matrix(rows, columns),
                dot,
                i,
                j,
                k;
            for (i = 0; i < rows; i++) {
                for (j = 0; j < columns; j++) {
                    dot = 0.0;
                    for (k = 0; k < matrix1.columns; k++) {
                        dot += matrix1.w[matrix1.columns * i + k] * matrix2.w[matrix2.columns * k + j];
                    }
                    out.w[columns * i + j] = dot;
                }
            }

            if (this.needsBackpropagation) {
                var backward = function () {
                    var outDw;
                    for (i = 0; i < rows; i++) {
                        for (j = 0; j < columns; j++) {
                            outDw = out.dw[columns * i + j];
                            for (k = 0; k < matrix1.columns; k++) {
                                matrix1.dw[matrix1.columns * i + k] += matrix2.w[matrix2.columns * k + j] * outDw;
                                matrix2.dw[matrix2.columns * k + j] += matrix1.w[matrix1.columns * i + k] * outDw;
                            }
                        }
                    }
                };
                this.backprop.push(backward);
            }
            return out;
        },
        add: function (matrix1, matrix2) {
            assert(matrix1.w.length === matrix2.w.length && matrix1.rows === matrix2.rows, 'matrix dimensions not compatible for addition');

            var out = new Matrix(matrix1.rows, matrix1.columns),
                i,
                n = matrix1.w.length;
            for (i = 0; i < n; i++) {
                out.w[i] = matrix1.w[i] + matrix2.w[i];
            }

            if (this.needsBackpropagation) {
                var backward = function () {
                    n = matrix1.w.length;
                    for (i = 0; i < n; i++) {
                        matrix1.dw[i] += out.dw[i];
                        matrix2.dw[i] += out.dw[i];
                    }
                };
                this.backprop.push(backward);
            }
            return out;
        },
        eltmul: function (matrix1, matrix2) {
            assert(matrix1.w.length === matrix2.w.length, 'matrix dimensions not compatible for element-multiplication');

            var out = new Matrix(matrix1.rows, matrix1.columns),
                i,
                n = matrix1.w.length;
            for (i = 0; i < n; i++) {
                out.w[i] = matrix1.w[i] * matrix2.w[i];
            }

            if (this.needsBackpropagation) {
                var backward = function () {
                    n = matrix1.w.length;
                    for (i = 0; i < n; i++) {
                        matrix1.dw[i] += matrix2.w[i] * out.dw[i];
                        matrix2.dw[i] += matrix1.w[i] * out.dw[i];
                    }
                };
                this.backprop.push(backward);
            }
            return out;
        }
    };

    // softmax function that squashes matrix values in a way that they range from 0 to 1 and add up to 1
    var softmax = function (matrix) {
        var out = new Matrix(matrix.rows, matrix.columns),
            maxVal = -Infinity,
            i,
            n = matrix.w.length;
        for (i = 0; i < n; i++) {
            if (matrix.w[i] > maxVal) {
                maxVal = matrix.w[i];
            }
        }

        var sum = 0.0;
        for (i = 0; i < n; i++) {
            out.w[i] = Math.exp(matrix.w[i] - maxVal);
            sum += out.w[i];
        }
        for (i = 0, n = matrix.w.length; i < n; i++) {
            out.w[i]  /= sum;
        }

        return out;
    };

    // performs one forward pass for a Recurrent Neural Network
    var forwardRNN = function (graph, model, previousNodes, hiddenSizes, sourceVector) {

        // sourceVector is a 1D vector with observations
        var previousHiddenNodes = [],
            depth;
        if (typeof previousNodes.hidden === 'undefined') {
            for (depth = 0; depth < hiddenSizes.length; depth++) {
                previousHiddenNodes.push(new Matrix(hiddenSizes[depth], 1));
            }
        } else {
            previousHiddenNodes = previousNodes.hidden;
        }

        var hiddenResults = [],
            inputVector,
            previousHiddenNode,
            hidden0,
            hidden1,
            hidden_depth;
        // perform operations for all hidden layers
        for (depth = 0; depth < hiddenSizes.length; depth++) {

            inputVector = (depth === 0 ? sourceVector : hiddenResults[depth - 1]);
            previousHiddenNode = previousHiddenNodes[depth];

            // take input
            hidden0 = graph.mul(model['Wxh' + depth], inputVector);
            // take state of previous hidden unit
            hidden1 = graph.mul(model['Whh' + depth], previousHiddenNode);

            // add hidden0, hidden1 and the hidden bias vector, then perform relu
            hidden_depth = graph.relu(graph.add(graph.add(hidden0, hidden1), model['bhh' + depth]));

            hiddenResults.push(hidden_depth);
        }

        // decode result of last hidden node to output
        var output = graph.add(graph.mul(model.Whd, hiddenResults[hiddenResults.length - 1]), model.bd);

        // return hidden representation and output
        return {'output': output, 'hidden': hiddenResults};

    };

    // performs one forward pass for a Gated Recurrent Unit
    var forwardGRU = function (graph, model, previousNodes, hiddenSizes, sourceVector) {

        var previousHiddenNodes = [],
            depth;
        if (typeof previousNodes.hidden === 'undefined') {
            for (depth = 0; depth < hiddenSizes.length; depth++) {
                previousHiddenNodes.push(new Matrix(hiddenSizes[depth], 1));
            }
        } else {
            previousHiddenNodes = previousNodes.hidden;
        }

        var hiddenResults = [],
            inputVector,
            previousHiddenNode,
            hidden0,
            hidden1,
            resetGate,
            hidden2,
            hidden3,
            updateGate,
            hidden4,
            hidden5,
            cell,
            hiddenAtDepth,
            allOnes,
            negUpdateGate;
        for (depth = 0; depth < hiddenSizes.length; depth++) {
            inputVector = (depth === 0 ? sourceVector : hiddenResults[depth - 1]);
            previousHiddenNode = previousHiddenNodes[depth];

            // reset gate
            hidden0 = graph.mul(model['Wrxh' + depth], inputVector);
            hidden1 = graph.mul(model['Wrhh' + depth], previousHiddenNode);
            resetGate = graph.sigmoid(graph.add(graph.add(hidden0, hidden1), model['br' + depth]));

            // update gate
            hidden2 = graph.mul(model['Wzxh' + depth], inputVector);
            hidden3 = graph.mul(model['Wzhh' + depth], previousHiddenNode);
            updateGate = graph.sigmoid(graph.add(graph.add(hidden2, hidden3), model['bz' + depth]));


            // cell
            hidden4 = graph.mul(model['Wcxh' + depth], inputVector);
            hidden5 = graph.mul(model['Wchh' + depth], graph.eltmul(resetGate, previousHiddenNode));
            cell = graph.tanh(graph.add(graph.add(hidden4, hidden5), model['bc' + depth]));

            // compute hidden state as gated, saturated cell activations
            allOnes = createMatrixFilledWithOnes(updateGate.rows, updateGate.columns);
            // negate updateGate
            negUpdateGate = createNegatedCloneMatrix(updateGate);
            hiddenAtDepth = graph.add(graph.eltmul(graph.add(allOnes, negUpdateGate), cell), graph.eltmul(previousHiddenNode, updateGate));

            hiddenResults.push(hiddenAtDepth);
        }

        // decode results of last hidden unit to output
        var output = graph.add(graph.mul(model.Whd, hiddenResults[hiddenResults.length - 1]), model.bd);

        // return hidden representation and output
        return {'output': output, 'hidden': hiddenResults};


    };

    // performs one forward pass for a Long Short-Term Memory
    var forwardLSTM = function (graph, model, previousNodes, hiddenSizes, sourceVector) {

        var previousHiddenNodes = [],
            previousCells = [],
            depth;
        if (typeof previousNodes.hidden === 'undefined') {
            for (depth = 0; depth < hiddenSizes.length; depth++) {
                previousHiddenNodes.push(new Matrix(hiddenSizes[depth], 1));
                previousCells.push(new Matrix(hiddenSizes[depth], 1));
            }
        } else {
            previousHiddenNodes = previousNodes.hidden;
            previousCells = previousNodes.cells;
        }

        var hiddenResults = [],
            cellResults = [],
            inputVector,
            previousHiddenNode,
            previousCell,
            hidden0,
            hidden1,
            inputGate,
            hidden2,
            hidden3,
            forgetGate,
            hidden4,
            hidden5,
            outputGate,
            hidden6,
            hidden7,
            cellWrite,
            retainCell,
            writeCell,
            cellAtDepth,
            hiddenAtDepth;
        for (depth = 0; depth < hiddenSizes.length; depth++) {
            inputVector = (depth === 0 ? sourceVector : hiddenResults[depth - 1]);
            previousHiddenNode = previousHiddenNodes[depth];
            previousCell = previousCells[depth];

            // input gate
            hidden0 = graph.mul(model['Wixh' + depth], inputVector);
            hidden1 = graph.mul(model['Wihh' + depth], previousHiddenNode);
            // add hidden0, hidden1 and the input bias vector, then perform sigmoid
            inputGate = graph.sigmoid(graph.add(graph.add(hidden0, hidden1), model['bi' + depth]));

            //forget gate
            hidden2 = graph.mul(model['Wfxh' + depth], inputVector);
            hidden3 = graph.mul(model['Wfhh' + depth], previousHiddenNode);
            // add hidden2, hidden3 and the forget bias vector, then perform sigmoid
            forgetGate = graph.sigmoid(graph.add(graph.add(hidden2, hidden3), model['bf' + depth]));

            // output gate
            hidden4 = graph.mul(model['Woxh' + depth], inputVector);
            hidden5 = graph.mul(model['Wohh' + depth], previousHiddenNode);
            // add hidden4, hidden5 and the output bias vector, then perform sigmoid
            outputGate = graph.sigmoid(graph.add(graph.add(hidden4, hidden5), model['bo' + depth]));

            // write operation on cells
            hidden6 = graph.mul(model['Wcxh' + depth], inputVector);
            hidden7 = graph.mul(model['Wchh' + depth], previousHiddenNode);
            cellWrite = graph.tanh(graph.add(graph.add(hidden6, hidden7), model['bc' + depth]));

            // compute new cell activation
            retainCell = graph.eltmul(forgetGate, previousCell); // what we keep from cell
            writeCell = graph.eltmul(inputGate, cellWrite); // what we write to cell
            cellAtDepth = graph.add(retainCell, writeCell); // new cell contents

            // compute hidden state as gated, saturated cell activations
            hiddenAtDepth = graph.eltmul(outputGate, graph.tanh(cellAtDepth));

            hiddenResults.push(hiddenAtDepth);
            cellResults.push(cellAtDepth);
        }

        // decode results of last hidden unit to output
        var output = graph.add(graph.mul(model.Whd, hiddenResults[hiddenResults.length - 1]), model.bd);

        // return cell memory, hidden representation and output
        return {'output': output, 'hidden': hiddenResults, 'cells': cellResults};


    };

    /*
     *   Main object visible from outside the library, that contains and handles graphs, matrices and
     *   other parameters for the neural networks.
     *   It offers multiple functions for user input and control.
     *   The weight matrices of the neural networks are saved in the model.
     */
    var NeuralNetwork = function (type) {
        if (typeof type === 'undefined' || type === 'LSTM') {
            this.type = 'LSTM';
        } else if (type === 'RNN') {
            this.type = 'RNN';
        } else if (type === 'GRU') {
            this.type = 'GRU';
        } else {
            throw new Error('Unknown type of NeuralNetwork');
        }

        this.model = {};
        this.graph = new Graph(true);
        this.previousNodes = {};

        this.predictGraph = new Graph(false);
        this.predictPreviousNodes = {};

        // variables for parameter update with default values
        this.stepCache = {};

    };

    NeuralNetwork.prototype = {
        // can be overridden by the user if he or she sets the value once
        decayRate: 0.999,
        smoothEps: 1e-8,
        clipValue: 5.0,
        regularizationConstant: 0.000001,

        // initializes the model and other parameters based on the type of the neural network
        initialize: function (inputSize, hiddenSizes, outputSize, options) {
            this.inputSize = inputSize;
            this.hiddenSizes = hiddenSizes;
            this.outputSize = outputSize;
            var depth,
                prevSize,
                hiddenSize;
            for (depth = 0; depth < hiddenSizes.length; depth++) {
                prevSize = (depth === 0 ? inputSize : hiddenSizes[depth - 1]);
                hiddenSize = hiddenSizes[depth];

                switch (this.type) {
                case 'RNN':
                    //input to hidden
                    this.model['Wxh' + depth] = createRandomizedMatrix(hiddenSize, prevSize, 0.08);
                    //hidden to hidden
                    this.model['Whh' + depth] = createRandomizedMatrix(hiddenSize, hiddenSize, 0.08);
                    // hidden bias vector
                    this.model['bhh' + depth] = new Matrix(hiddenSize, 1);
                    break;

                case 'GRU':
                    // reset Gate
                    this.model['Wrxh' + depth] = createRandomizedMatrix(hiddenSize, prevSize, 0.08);
                    this.model['Wrhh' + depth] = createRandomizedMatrix(hiddenSize, hiddenSize, 0.08);
                    this.model['br' + depth] = new Matrix(hiddenSize, 1);
                    // update Gate
                    this.model['Wzxh' + depth] = createRandomizedMatrix(hiddenSize, prevSize, 0.08);
                    this.model['Wzhh' + depth] = createRandomizedMatrix(hiddenSize, hiddenSize, 0.08);
                    this.model['bz' + depth] = new Matrix(hiddenSize, 1);
                    // cell write parameters
                    this.model['Wcxh' + depth] = createRandomizedMatrix(hiddenSize, prevSize, 0.08);
                    this.model['Wchh' + depth] = createRandomizedMatrix(hiddenSize, hiddenSize, 0.08);
                    this.model['bc' + depth] = new Matrix(hiddenSize, 1);
                    break;

                case 'LSTM':
                    // gate parameters
                    // input gate: - input to hidden, hidden to hidden and bias vector
                    this.model['Wixh' + depth] = createRandomizedMatrix(hiddenSize, prevSize, 0.08);
                    this.model['Wihh' + depth] = createRandomizedMatrix(hiddenSize, hiddenSize, 0.08);
                    this.model['bi' + depth] = new Matrix(hiddenSize, 1);
                    // forget gate: input to hidden, hidden to hidden and bias vector
                    this.model['Wfxh' + depth] = createRandomizedMatrix(hiddenSize, prevSize, 0.08);
                    this.model['Wfhh' + depth] = createRandomizedMatrix(hiddenSize, hiddenSize, 0.08);
                    this.model['bf' + depth] = new Matrix(hiddenSize, 1);
                    // output gate: input to hidden, hidden to hidden and bias vector
                    this.model['Woxh' + depth] = createRandomizedMatrix(hiddenSize, prevSize, 0.08);
                    this.model['Wohh' + depth] = createRandomizedMatrix(hiddenSize, hiddenSize, 0.08);
                    this.model['bo' + depth] = new Matrix(hiddenSize, 1);

                    // cell write parameters
                    this.model['Wcxh' + depth] = createRandomizedMatrix(hiddenSize, prevSize, 0.08);
                    this.model['Wchh' + depth] = createRandomizedMatrix(hiddenSize, hiddenSize, 0.08);
                    this.model['bc' + depth] = new Matrix(hiddenSize, 1);
                    break;

                default:
                    throw new Error('Unknown type of NeuralNetwork');
                }

            }
            // decoder parameters
            this.model.Whd = createRandomizedMatrix(outputSize, hiddenSize, 0.08);
            this.model.bd = new Matrix(outputSize, 1);

            // options
            if (typeof options !== 'undefined') {
                if (options.clipValue !== 'undefined') {
                    this.clipValue = options.clipValue;
                }
                if (options.regularizationConstant !== 'undefined') {
                    this.regularizationConstant = options.regularizationConstant;
                }
            }

        },
        // performs one forward pass, based on the type of neural network
        forward: function (graph, sourceVector) {
            if (typeof graph === 'undefined') { graph = this.graph}
            var forward = {};
            switch (this.type) {
            case 'RNN':
                forward = forwardRNN(graph, this.model, this.previousNodes, this.hiddenSizes, sourceVector);
                break;
            case 'GRU':
                forward = forwardGRU(graph, this.model, this.previousNodes, this.hiddenSizes, sourceVector);
                break;
            case 'LSTM':
                forward = forwardLSTM(graph, this.model, this.previousNodes, this.hiddenSizes, sourceVector);
                break;
            default:
                throw new Error('Unknown type of NeuralNetwork');
            }
            this.previousNodes = forward;

            return forward;
        },
        /*
         *   performs the update of all weight matrices AFTER the backpropagation has been done
         *   and resets the .dw arrays of all matrices
         *   also resets the predictionGraph and predictionPreviousNodes
         */
        parameterUpdate: function (learningRate, regularizationConstant, clipValue) {
            var statistics = {},
                numberClipped = 0,
                numberTotalOperations = 0,
                key,
                matrix,
                stepCacheMatrix,
                i,
                matrixDwi,
                n;
            if (typeof regularizationConstant !== 'undefined') {
                this.regularizationConstant = regularizationConstant;
            }
            if (typeof clipValue !== 'undefined') {
                this.clipValue = clipValue;
            }
            for (key in this.model) {
                if (this.model.hasOwnProperty(key)) {
                    matrix = this.model[key];
                    if (!(key in this.stepCache)) {
                        this.stepCache[key] = new Matrix(matrix.rows, matrix.columns);
                    }
                    stepCacheMatrix = this.stepCache[key];
                    n = matrix.w.length;
                    for (i = 0; i < n; i++) {

                        // rmsprop adaptive learning rate
                        matrixDwi = matrix.dw[i];
                        stepCacheMatrix.w[i] = stepCacheMatrix.w[i] * this.decayRate + (1.0 - this.decayRate) * matrixDwi * matrixDwi;

                        // gradient clip
                        if (matrixDwi > this.clipValue) {
                            matrixDwi = this.clipValue;
                            numberClipped++;
                        }
                        if (matrixDwi < -this.clipValue) {
                            matrixDwi = -this.clipValue;
                            numberClipped++;
                        }
                        numberTotalOperations++;

                        // update and regularize
                        matrix.w[i] += -learningRate * matrixDwi / Math.sqrt(stepCacheMatrix.w[i] + this.smoothEps) - this.regularizationConstant * matrix.w[i];
                        matrix.dw[i] = 0; // reset gradients for next iteration
                    }
                }
            }
            statistics.ratioClipped = numberClipped / numberTotalOperations;

            /*
             *   When a network is newly trained, the old graphs and nodes should not influence the new ones, so they are
             *   completely reset here.
             */
            this.predictPreviousNodes = {};
            this.predictGraph = new Graph(false);

            return statistics;
        },
        /*
         *  performs a forward pass using a predictGraph that has backpropagation turned off, then performs softmax
         *  and, depending on user input, either returns most probable output index or a sample output index
         */
        predictOutput: function (sourceVector, useSampling, temperature) {

            function searchMaximumIndex(w) {
                // argmax of array w
                var maxValue = w[0],
                    maxIndex = 0,
                    i,
                    n = w.length;
                for (i = 1; i < n; i++) {
                    if (w[i] > maxValue) {
                        maxIndex = i;
                        maxValue = w[i];
                    }
                }
                return maxIndex;
            }

            function sampleIndex(w) {
                // sample argmax from w, assuming w are probabilities
                // that sum to one
                var r = Math.random(),
                    x = 0.0,
                    i;
                for (i = 0; i < w.length; i++) {
                    x += w[i];
                    if (x > r) {
                        break;
                    }
                }
                return i;
            }
            if (typeof useSampling === 'undefined') {
                useSampling = false;
            }
            if (typeof temperature === 'undefined') {
                temperature = 1.0;
            }
            var forward = {};
            switch (this.type) {
            case 'RNN':
                forward = forwardRNN(this.predictGraph, this.model, this.predictPreviousNodes, this.hiddenSizes, sourceVector);
                break;
            case 'GRU':
                forward = forwardGRU(this.predictGraph, this.model, this.predictPreviousNodes, this.hiddenSizes, sourceVector);
                break;
            case 'LSTM':
                forward = forwardLSTM(this.predictGraph, this.model, this.predictPreviousNodes, this.hiddenSizes, sourceVector);
                break;
            default:
                throw new Error('Unknown type of NeuralNetwork');
            }
            this.predictPreviousNodes = forward;
            var logProbabilities = forward.output;


            if (temperature !== 1.0 && useSampling) {
                // scale log probabilities by temperature
                // if the temperature is high, log probabilities will go towards zero
                // and the softmax output will be more diffuse, otherwise it will be more peaky
                var q,
                    n = logProbabilities.w.length;
                for (q = 0; q < n; q++) {
                    logProbabilities.w[q] /= temperature;
                }
            }

            var probabilities = softmax(logProbabilities);
            var index;
            if (useSampling) {
                index = sampleIndex(probabilities.w);
            } else {
                index = searchMaximumIndex(probabilities.w);
            }
            return index;

        },
        // performs backpropagation, clears the backpropagate array of the graph
        backward: function () {
            this.graph.performBackpropagation();
        },
        /*
         *  performs a forward pass by calling the forward function of the Neural Network, then calculates
         *  and returns perplexity and cost. Also sets some .dw matrices in preparation of the backpropagation
         */

        costFunction: function (sourceVector, targetIndex) {
            var logToPerplexity,
                cost,
                graph = this.graph,
                forward = this.forward(graph, sourceVector),
                logProbabilities = forward.output,
                probabilities = softmax(logProbabilities),
                totalPerplexity;

            logToPerplexity = -Math.log2(probabilities.w[targetIndex]); // accumulate base 2 log prob and do smoothing
            cost = -Math.log(probabilities.w[targetIndex]);

            // write gradients into .dw matrices - for backpropagation
            logProbabilities.dw = probabilities.w;
            logProbabilities.dw[targetIndex] -= 1;

            totalPerplexity = Math.pow(2, logToPerplexity);
            return {'cost': cost, 'perplexity': logToPerplexity, 'totalPerplexity': totalPerplexity};
        },

        /*
         *  performs the costFunction for a batch of input and calculates the corresponding perplexity and cost
         */
        batchCostFunction: function (sourceVectorArray, targetIndexArray) {
            var costReturn,
                cost = 0.0,
                logToPerplexity = 0.0,
                i,
                n = sourceVectorArray.length,
                totalPerplexity;
            for (i = 0; i < n; i++) {
                costReturn = this.costFunction(sourceVectorArray[i], targetIndexArray[i]);
                cost += costReturn.cost;
                logToPerplexity += costReturn.perplexity;
                console.log(costReturn.perplexity + "| " + costReturn.totalPerplexity);
            }
            totalPerplexity = Math.pow(2, logToPerplexity / (n - 1));
            return {'perplexity': totalPerplexity, 'cost': cost}
        },
        /*
         *  Utility functions for user convenience. If the user does not want to create his or her own input vectors of
         *  the right size, generateInputMatrix will create a randomized matrix that will allow the user to just use
         *  an index and the getInputVector function as input vectors.
         */
        generateInputMatrix: function (diversificationSize) {
            this.model.WInput = createRandomizedMatrix(diversificationSize, this.inputSize, 0.08);
            return this.model.WInput;
        },
        getInputVector: function (sourceIndex) {
            var g = new Graph();
            return g.pluckRow(this.model.WInput, sourceIndex);
        }

    };

    // only the NeuralNetwork will be visible from the outside
    globalAccess.NeuralNetwork = NeuralNetwork;

})(Neuraljs);