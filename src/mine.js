var Neuraljs = {};

(function(globalAccess) {
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

    var gaussReturn = false;
    var gaussValueV = 0.0;

    var gaussRandom = function() {
        if (gaussReturn) {
            gaussReturn = false;
            return gaussValueV;
        }
        var u = 2*Math.random()-1;
        var v = 2*Math.random()-1;

        var r = u*u + v*v;

        if (r === 0 || r > 1) return gaussRandom();

        var c = Math.sqrt(-2*Math.log(r)/r);
        gaussValueV = v*c; // cached for return on next call
        gaussReturn = true;
        return u*c;
    };

    var zeros = function (n) {
        if (typeof(n) === 'undefined' || isNaN(n)) { return []; }
        if (typeof ArrayBuffer === 'undefined') {
            // no browser support
            var arr = [];
            for (var i = 0; i < n; i++) { arr[i] = 0; }
            return arr;
        }
        return new Float64Array(n);
    };

    var Matrix = function(rows, columns) {
        if (typeof rows === 'undefined' || typeof columns === 'undefined') {
            throw new Error('Undefined not allowed in Matrix initialization')
        }
        this.rows = rows;
        this.columns = columns;
        this.w = zeros(rows * columns);
        this.dw = zeros(rows * columns);
    };
    Matrix.prototype = {
        get: function(row, column) {
            var index = (this.rows * row) + column;
            assert((index >= 0) && (index < this.w.length));
            return this.w[index];
        },
        set: function(row, column, value) {
            var index = (this.rows * row) + column;
            assert((index >= 0) && (index < this.w.length));
            this.w[index] = value;
        },
        toJSON: function() {
            var json = {};
            json['rows'] = this.rows;
            json['columns'] = this.columns;
            json['w'] = this.w;
            return json;
        },
        fromJSON: function(json){
            this.rows = json.rows;
            this.columns = json.columns;
            this.w = zeros(this.rows * this.columns);
            this.dw = zeros(this.rows * this.columns);
            for (var i = 0, n = this.rows * this.columns; i < n; i++) {
                this.w[i] = json.w[i];
            }
        }
    };

    var RandomMatrix = function(rows, columns, std) {

        var fillRandom = function(matrix, low, high) {
            for(var i = 0, n = matrix.rows * matrix.columns; i < n; i++) {
                matrix.w[i] = Math.random()*(high-low)+low;
            }
        };

        var m = new Matrix(rows, columns);
        fillRandom(m, -std,std);
        return m;

    }

    /*
    Class responsible for mathematical operations on matrices. It automatically saves the
    necessary backpropagation functions, which will update the dw array of the matrix
     */
    var Graph = function(needsBackpropagation) {
        if (typeof needsBackpropagation === 'undefined') { needsBackpropagation = true; }
        this.needsBackpropagation = needsBackpropagation;

        // this stores a list of functions that perform the backpropagation in the right order
        this.backprop = [];
    }
    Graph.prototype = {
        performBackpropagation: function() {
            for ( var i = this.backprop.length - 1; i <= 0; i--) {
                this.backprop[i]; // execution of one function
            }
        },
        pluckRow: function(matrix, rowIndex) {
            // returns the row as a column vector
            assert((rowIndex >= 0) && (rowIndex <= matrix.rows));
            var columns = matrix.columns;
            var out = new Matrix(columns, 1);
            for (var i = 0; i < columns; i++) {
                out.w[i] = matrix.w[columns * rowIndex + i];
            }

            if (this.needsBackpropagation) {
                var backward = function() {
                    for (var i = 0; i < columns; i++) {
                        matrix.dw[columns * rowIndex + i] += out.dw[i];
                    }
                    this.backprop.push(backward);
                }
            }
            return out;
        },
        tanh: function(matrix) {
            var out = new Matrix(matrix.rows, matrix.columns);
            var n = matrix.w.length;
            for (var i = 0; i < n; i++) {
                out.w[i] = Math.tanh(matrix.w[i]);
            }

            if (this.needsBackpropagation) {
                var backward = function() {
                    var matrixWi = out.w[i];
                    matrix.dw[i] += (1.0 - matrixWi * matrixWi) * out.dw[i];
                }
                this.backprop.push(backward);
            }
            return out;
        },
        sigmoid: function(matrix) {
            var out = new Matrix(matrix.rows, matrix.columns);
            var n = matrix.w.length;
            for (var i = 0; i < n; i++) {
                out.w[i] = 1.0/(1+Math.exp(-matrix.w[i]));
            }

            if (this.needsBackpropagation) {
                var backward = function() {
                    for (var i = 0; i < n; i++) {
                        var matrixWi = out.w[i];
                        m.dw[i] += matrixWi * (1.0 - matrixWi) * out.dw[i];
                    }
                }
                this.backprop.push(backward);
            }
            return out;
        },
        relu: function(matrix) {
            var out = new Matrix(matrix.rows, matrix.columns);
            var n = matrix.w.length;
            for (var i = 0; i < n; i++) {
                out.w[i] = Math.max(0, matrix.w[i]);
            }

            if (this.needsBackpropagation) {
                var backward = function() {
                    for (var i = 0; i < n; i++) {
                        matrix.dw[i] += matrix.w[i] > 0 ? out.dw[i] : 0.0;
                    }
                }
                this.backprop.push(backward);
            }
            return out;
        },
        mul: function(matrix1, matrix2) {
            assert(matrix1.columns === matrix2.rows, 'matrix dimensions not compatible for multiplication');

            var rows = matrix1.rows;
            var columns = matrix2.columns;
            var out = new Matrix(rows, columns);
            var dot;
            for (var i = 0; i < rows; i++) {
                for (var j = 0; j < columns; j++) {
                    dot = 0.0;
                    for (var k = 0; k < matrix1.columns; k++) {
                        dot += matrix1.w[matrix1.columns * i + k] * matrix2.w[matrix2.columns * k + j];
                    }
                    out.w[columns * i + j] = dot;
                }
            }

            if (this.needsBackpropagation) {
                var backward = function() {
                    var help;
                    for (i = 0; i < rows; i++) {
                        for (j = 0; j < columns; j++) {
                            help = out.dw[columns * i + j];
                            for (k = 0; k < matrix1.columns; k++) {
                                matrix1.dw[matrix1.columns * i + k] += matrix2.w[matrix2.columns * k + j] * help;
                                matrix2.dw[matrix2.columns * k + j] += matrix1.w[matrix1.columns * i + k] * help;
                            }
                        }
                    }
                }
                this.backprop.push(backward);
            }
            return out;
        },
        add: function(matrix1, matrix2) {
            assert(matrix1.w.length === matrix2.w.length && matrix1.rows === matrix2.rows, 'matrix dimensions not compatible for addition');

            var out = new Matrix(matrix1.rows, matrix1.columns);
            for (var i = 0; i < matrix1.w.length; i++) {
                out.w[i] = matrix1.w[i] + matrix2.w[i];
            }

            if (this.needsBackpropagation) {
                var backward = function() {
                    for (var i = 0; i < matrix1.w.length; i++) {
                        matrix1.dw[i] += matrix2.w[i] * out.dw[i];
                        matrix2.dw[i] += matrix1.w[i] * out.dw[i];
                    }
                }
                this.backprop.push(backward);
            }
            return out;
        },
        eltmul: function(matrix1, matrix2) {
            assert(matrix1.w.length === matrix2.w.length, 'matrix dimensions not compatible for element-multiplication');

            var out = new Matrix(matrix1.rows, matrix1.columns);
            for (var i = 0; i < matrix1.w.length; i++) {
                out.w[i] = matrix1.w[i] * matrix2.w[i];
            }

            if (this.needsBackpropagation) {
                var backward = function() {
                    for (var i = 0; i < matrix1.w.length; i++) {
                        matrix1.dw[i] += matrix2.w[i] * out.dw[i];
                        matrix2.dw[i] += matrix1.w[i] * out.dw[i];
                    }
                }
                this.backprop.push(backward);
            }
            return out;
        },
    }

    var softmax = function(matrix) {
        var out = new Matrix(matrix.rows, matrix.columns);
        var maxVal = -Infinity;
        for (var i = 0, n = matrix.w.length; i < n; i++) {
            if (matrix.w[i] > maxVal) maxVal = matrix.w[i];
        }

        var sum = 0.0;
        for (i = 0,n = matrix.w.length; i < n; i++) {
            out.w[i] = Math.exp(matrix.w[i] - maxVal);
            sum += out.w[i];
        }
        for (i = 0, n = matrix.w.length; i < n; i++) {
            out.w[i]  /= sum;
        }

        return out;
    }

    var forwardRNN = function(graph, model, previousNodes, hiddenSizes, sourceVector){
        // sourceVector is a 1D vector with observations
        var previousHiddenNodes = [];
        if (typeof previousNodes.hidden === 'undefined') {
            for (var depth = 0; depth < hiddenSizes.length; depth++) {
                previousHiddenNodes.push(new Matrix(hiddenSizes[depth],1));
            }
        } else {
            previousHiddenNodes = previousNodes.hidden;
        }

        var hiddenResults = [];
        for (var depth = 0; depth < hiddenSizes.length; depth++) {

            var inputVector = (depth === 0 ? sourceVector : hiddenResults[depth - 1]);
            var previousHiddenNode = previousHiddenNodes[depth];

            // take input
            var hidden0 = graph.mul(model['Wxh' + depth], inputVector);
            // take state of previous hidden unit
            var hidden1 = graph.mul(model['Whh' + depth], previousHiddenNode);

            // add hidden0, hidden1 and the hidden bias vector, then perform relu
            var hidden_depth = graph.relu(graph.add(graph.add(hidden0, hidden1), model['bhh' + depth]));

            hiddenResults.push(hidden_depth);
        }

        // decode result of last hidden node to output
        var output = graph.add(graph.mul(model['Whd'], hiddenResults[hiddenResults.length - 1]), model['bd']);

        // return hidden representation and output
        return {'output': output, 'hidden': hiddenResults};

        // TODO: original: return {'h':hiddenResults, 'o': output};

    };

    var forwardGRU = function(graph, model, previousNodes, hiddenSizes, sourceVector){

    };

    var forwardLSTM = function(graph, model, previousNodes, hiddenSizes, sourceVector){
        var previousHiddenNodes = [];
        var previousCells = [];
        if (typeof previousNodes.hidden === 'undefined') {
            for (var depth = 0; depth < hiddenSizes.length; depth++) {
                previousHiddenNodes.push( new Matrix(hiddenSizes[depth], 1));
                previousCells.push(new Matrix(hiddenSizes[depth],1));
            }
        } else {
            previousHiddenNodes = previousNodes.hidden;
            previousCells = previousNodes.cells;
        }

        var hiddenResults = [];
        var cellResults = [];
        for (var depth = 0; depth < hiddenSizes.length; depth++) {
            var inputVector = ( depth === 0 ? sourceVector : hiddenResults[depth-1]);
            var previousHiddenNode = previousHiddenNodes[depth];
            var previousCell = previousCells[depth];

            // input gate
            var hidden0 = graph.mul(model['Wixh' + depth], inputVector);
            var hidden1 = graph.mul(model['Wihh' + depth], previousHiddenNode);
            // add hidden0, hidden1 and the input bias vector, then perform sigmoid
            var inputGate = graph.sigmoid(graph.add(graph.add(hidden0, hidden1), model['bi' + depth]));

            //forget gate
            var hidden2 = graph.mul(model['Wfxh' + depth], inputVector);
            var hidden3 = graph.mul(model['Wfhh' + depth], previousHiddenNode);
            // add hidden2, hidden3 and the forget bias vector, then perform sigmoid
            var forgetGate = graph.sigmoid(graph.add(graph.add(hidden2, hidden3), model['bf' + depth]));

            // output gate
            var hidden4 = graph.mul(model['Woxh' + depth], inputVector);
            var hidden5 = graph.mul(model['Wohh' + depth], previousHiddenNode);
            // add hidden4, hidden5 and the output bias vector, then perform sigmoid
            var outputGate = graph.sigmoid(graph.add(graph.add(hidden4, hidden5), model['bo' + depth]));

            // write operation on cells
            var hidden6 = graph.mul(model['Wcxh' + depth], inputVector);
            var hidden7 = graph.mul(model['Wchh' + depth], previousHiddenNode);
            var cellWrite = graph.tanh(graph.add(graph.add(hidden6, hidden7), model['bc' + depth]));

            // compute new cell activation
            var retainCell = graph.eltmul(forgetGate, previousCell); // what we keep from cell
            var writeCell = graph.eltmul(inputGate, cellWrite); // what we write to cell
            var cellAtDepth = graph.add(retainCell, writeCell); // new cell contents

            // compute hidden state as gated, saturated cell activations
            var hiddenAtDepth = graph.eltmul(outputGate, graph.tanh(cellAtDepth));

            hiddenResults.push(hiddenAtDepth);
            cellResults.push(cellAtDepth);
        }

        // decode results of last hidden unit to output
        var output = graph.add(graph.mul(model['Whd'], hiddenResults[hiddenResults.length - 1]), model['bd']);

        // return cell memory, hidden representation and output
        return {'output': output, 'hidden': hiddenResults, 'cells':cellResults}


    };


    var NeuralNetwork = function(type) {
        if (typeof type === 'undefined' || type === 'LSTM') {
            this.type = 'LSTM';
        }
        else if (type === 'RNN') {
            this.type = 'RNN';
        }
        else if (type === 'GRU') {
            this.type = 'GRU';
        } else {
            throw new Error('Unknown type of NeuralNetwork');
        }

        this.model = {};
        this.graph = new Graph(true);
        this.previousNodes = {};
        this.predictPreviousNodes = {};
        this.predictGraph = new Graph(false);

        // variables for parameter update with default values
        this.decayRate = 0.999;
        this.smoothEps = 1e-8;
        this.stepCache = {};
        this.clipValue = 5.0;
        this.regularizationConstant = 0.000001;

    };

    NeuralNetwork.prototype = {
        initialize: function(inputSize, hiddenSizes, outputSize, options) {
            this.inputSize = inputSize;
            this.hiddenSizes = hiddenSizes;
            this.outputSize = outputSize;
            for (var depth = 0; depth < hiddenSizes.length; depth++) {
                var prevSize = (depth === 0 ? inputSize : hiddenSizes[depth - 1]);
                var hiddenSize = hiddenSizes[depth];

                switch (this.type) {
                    case 'RNN':
                        //input to hidden
                        this.model['Wxh' + depth] = new RandomMatrix(hiddenSize, prevSize, 0.08);
                        //hidden to hidden
                        this.model['Whh' + depth] = new RandomMatrix(hiddenSize, hiddenSize, 0.08);
                        // hidden bias vector
                        this.model['bhh' + depth] = new Matrix(hiddenSize, 1, 0.08);
                        break;

                    case 'GRU':
                        // TODO: to be implemented
                        break;

                    case 'LSTM':
                        // gates parameters
                        // input gate: - input to hidden, hidden to hidden and bias vector
                        this.model['Wixh' + depth] = new RandomMatrix(hiddenSize, prevSize, 0.08);
                        this.model['Wihh' + depth] = new RandomMatrix(hiddenSize, hiddenSize, 0.08);
                        this.model['bi' + depth] = new Matrix(hiddenSize, 1);
                        // forget gate: input to hidden, hidden to hidden and bias vector
                        this.model['Wfxh' + depth] = new RandomMatrix(hiddenSize, prevSize, 0.08);
                        this.model['Wfhh' + depth] = new RandomMatrix(hiddenSize, hiddenSize, 0.08);
                        this.model['bf' + depth] = new Matrix(hiddenSize, 1);
                        // output gate: input to hidden, hidden to hidden and bias vector
                        this.model['Woxh' + depth] = new RandomMatrix(hiddenSize, prevSize, 0.08);
                        this.model['Wohh' + depth] = new RandomMatrix(hiddenSize, hiddenSize, 0.08);
                        this.model['bo' + depth] = new Matrix(hiddenSize, 1);

                        // cell write parameters
                        this.model['Wcxh' + depth] = new RandomMatrix(hiddenSize, prevSize, 0.08);
                        this.model['Wchh' + depth] = new RandomMatrix(hiddenSize, hiddenSize, 0.08);
                        this.model['bc' + depth] = new Matrix(hiddenSize, 1);

                        break;

                    default:
                        throw new Error('Unknown type of NeuralNetwork');
                }

            }
            // decoder parameters
            this.model['Whd'] = new RandomMatrix(outputSize, hiddenSize, 0.08);
            this.model['bd'] = new Matrix(outputSize, 1);

            // options
            if (typeof options != 'undefined') {
                if (options['clipValue'] != 'undefined') {
                    this.clipValue = options['clipValue'];
                }
                if (options['regularizationConstant'] != 'undefined') {
                    this.regularizationConstant = options['regularizationConstant'];
                }
            }

        },
        forward: function(graph, sourceVector) {
            if (typeof graph === 'undefined') { graph = this.graph};
            var forward = {};
            switch(this.type) {
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
        parameterUpdate: function(learningRate, regularizationConstant, clipValue) {
            var statistics = {};
            var numberClipped = 0;
            var numberTotalOperations = 0;
            if (typeof regularizationConstant != 'undefined') {
                this.regularizationConstant = regularizationConstant;
            }
            if (typeof clipValue != 'undefined') {
                this.clipValue = clipValue;
            }
            for (var key in this.model) {
                if (this.model.hasOwnProperty(key)) {
                    var matrix = this.model[key];
                    if (!(key in this.stepCache)) {
                        this.stepCache[key] = new Matrix(matrix.rows, matrix.columns);
                    }
                    var s = this.stepCache[key];
                    for (var i = 0, n = matrix.w.length; i < n; i++) {

                        // rmsprop adaptive learning rate
                        var matrixDwi = matrix.dw[i];
                        s.w[i] = s.w[i] * this.decayRate + (1.0 - this.decayRate) * matrixDwi * matrixDwi;

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
                        matrix.w[i] += - learningRate * matrixDwi / Math.sqrt(s.w[i] + this.smoothEps) - this.regularizationConstant * matrix.w[i];
                        matrix.dw[i] = 0; // reset gradients for next iteration
                    }
                }
            }
            statistics['ratioClipped'] = numberClipped / numberTotalOperations;

            // TODO: explain the reset of the predict values here or do it differently - this is not intuitive!
            // reset predict Nodes after a the network is newly trained
            this.predictPreviousNodes = {};
            this.predictGraph = new Graph(false);

            return statistics;
        },
        predictOutput: function(sourceVector, temperature, useSamplei) { // TODO: think about input and output
            if (typeof useSamplei === 'undefined') { useSamplei = false; }
            if (typeof temperature === 'undefined') { temperature = 1.0; }
            var forward = {};
            switch(this.type) {
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
            // prev = lh; // TODO:
            var logProbabilities = forward.output;


            if (temperature !== 1.0 && useSamplei) {
                // scale log probabilities by temperature
                // if the temperature is high, log probabilities will go towards zero
                // and the softmax output will be more diffuse, otherwise it will be more peaky
                for (var q = 0, n = logProbabilities.w.length; q < n; q++) {
                    logprobs.w[q] /= temperature;
                }
            }

            var probabilities = softmax(logProbabilities);
            var index;
            if (useSamplei) {
                index = samplei(probabilities.w);
            } else {
                index = maxi(probabilities.w);
            }

            return index;

        },
        backward: function() {
          this.graph.performBackpropagation();
        },
        costFunction: function(sourceVector, targetVector, targetIndex) {
            var logToPerplexity = 0.0;
            var cost = 0.0;
            var graph = this.graph;
            var forward = this.forward(graph, sourceVector);
            // prev =
            var logProbabilities = forward.output;
            var probabilities = softmax(logProbabilities);

            // TODO: refactor so targetIndex does not have to be input?
            logToPerplexity += Math.log2(probabilities.w[targetIndex]); // accumulate base 2 log prob and do smoothing
            cost += -Math.log(probabilities.w[targetIndex]);

            // write gradients into logProbabilities;
            logProbabilities.dw = probabilities.w;
            logProbabilities.dw[targetIndex] -= 1;

            return {'cost':cost, 'perplexity':logToPerplexity};
        },
        generateInputMatrix: function(diversificationSize) {
            this.model['WInput'] = new RandomMatrix(diversificationSize, this.inputSize, 0.08);
            return this.model.WInput;
        },
        getInputVector: function(sourceIndex) {
            var g = new Graph;
            return g.pluckRow(this.model['WInput'],sourceIndex);
        },

    }

    var maxi = function(w) {
        // argmax of array w
        var maxValue = w[0];
        var maxIndex = 0;
        for (var i = 1, n = w.length; i < n; i++) {
            if (w[i] > maxValue) {
                maxIndex = i;
                maxValue = w[i];
            }
        }
        return maxIndex;
    };

    var samplei = function(w) {
        // sample argmax from w, assuming w are probabilities
        // that sum to one
        var r = Math.random();
        var x = 0.0;
        var i = 0;
        while(true) {
            x += w[i];
            if (x > r) {return i}
            i++;
        }
    };

    globalAccess.NeuralNetwork = NeuralNetwork;

})(Neuraljs);