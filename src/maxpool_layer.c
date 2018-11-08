#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    int w_i, h_i, im_i, c_i, x, y;
    int start = (l.size - 1) / 2;
    for (im_i = 0; im_i < in.rows; im_i++) {
        float *im_i_data = &in.data[im_i * in.cols];
        float *out_im_i_data = &out.data[im_i * out.cols];
        for (c_i = 0; c_i < l.channels; c_i++) {
            float *chan_i = &im_i_data[c_i * l.height * l.width];
            float *out_chan_i = &out_im_i_data[c_i * outh * outw];
            for (h_i = -start; h_i < l.height - start; h_i += l.stride) { // runs outh times
                for (w_i = -start; w_i < l.width - start; w_i += l.stride) { // runs outw times
                    float max = -FLT_MAX;
                    for (x = 0; x < l.size; x++) {
                        for (y = 0; y < l.size; y++) {
                            int col = x + w_i;
                            int row = y + h_i;
                            if (col < 0 || row < 0 || col >= l.width || row >= l.height) {
                                continue;
                            }
                            float curr = chan_i[row * l.width + col];
                            max = curr > max ? curr : max;
                        }
                    }
                    int curr_out_col = (w_i + start) / l.stride;
                    int curr_out_row = (h_i + start) / l.stride;
                    out_chan_i[curr_out_row * outw + curr_out_col] = max;
                }
            }
        }
    }

    l.in[0] = in;
    free_matrix(l.out[0]);
    l.out[0] = out;
    free_matrix(l.delta[0]);
    l.delta[0] = make_matrix(out.rows, out.cols);
    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix prev_delta: error term for the previous layer
void backward_maxpool_layer(layer l, matrix prev_delta)
{
    matrix in    = l.in[0];
    matrix out   = l.out[0];
    matrix delta = l.delta[0];

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;

    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    int w_i, h_i, im_i, c_i, x, y;
    int start = (l.size - 1) / 2;
    for (im_i = 0; im_i < in.rows; im_i++) {
        float *im_i_data = &in.data[im_i * in.cols];
        float *prev_delta_im_i_data = &prev_delta.data[im_i * prev_delta.cols];
        float *delta_im_i_data = &delta.data[im_i * delta.cols];
        for (c_i = 0; c_i < l.channels; c_i++) {
            float *chan_i = &im_i_data[c_i * l.height * l.width];
            float *prev_delta_chan_i = &prev_delta_im_i_data[c_i * l.height * l.width];
            float *delta_chan_i = &delta_im_i_data[c_i * outh * outw];
            for (h_i = -start; h_i < l.height - start; h_i += l.stride) { // runs outh times
                for (w_i = -start; w_i < l.width - start; w_i += l.stride) { // runs outw times
                    float max = -FLT_MAX;
                    int max_col = -1;
                    int max_row = -1;
                    for (x = 0; x < l.size; x++) {
                        for (y = 0; y < l.size; y++) {
                            int col = x + w_i;
                            int row = y + h_i;
                            if (col < 0 || row < 0 || col >= l.width || row >= l.height) {
                                continue;
                            }
                            float curr = chan_i[row * l.width + col];
                            if (curr > max) {
                                max = curr;
                                max_col = col;
                                max_row = row;
                            }
                        }
                    }
                    int curr_out_col = (w_i + start) / l.stride;
                    int curr_out_row = (h_i + start) / l.stride;
                    prev_delta_chan_i[max_row * l.width + max_col] += delta_chan_i[curr_out_row * outw + curr_out_col];
                }
            }
        }
    }
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay)
{
}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.in = calloc(1, sizeof(matrix));
    l.out = calloc(1, sizeof(matrix));
    l.delta = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

