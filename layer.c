#include "layer.h"

/* The sigmoid function and derivative. */
double sigmoid(double x)
{
  return 1/(1 +  exp(-x));
}

double sigmoidprime(double x)
{
  return x*(1 - x);
}

/* Creates a single layer. */
layer_t *layer_create()
{
  return (layer_t *) calloc(1, sizeof(layer_t));

}

/* Initialises the given layer. */
bool layer_init(layer_t *layer, int num_outputs, layer_t *prev)
{
  if (!layer) {
    perror("Given layer is NULL");
    return true;
  }
  layer -> num_outputs = num_outputs;
  layer-> prev = prev;
  layer -> outputs = (double *) calloc(num_outputs, sizeof(double));
  if (!layer -> outputs) {
    perror("Memory allocation error");
    return true;
  }

  if (prev) {
    layer -> num_inputs = prev -> num_outputs;
    layer -> biases = (double *) calloc(num_outputs, sizeof(double));
    layer -> deltas = (double *) calloc(num_outputs, sizeof(double));
    layer -> weights = (double **) calloc(layer -> num_inputs, sizeof(double *));
    if ((!layer -> biases) || (!layer -> deltas) || (!layer -> weights)) {
      perror("Memory allocation error");
      return true;
    }

    for (int i = 0; i < layer -> num_inputs; i++){
      layer -> weights[i] = (double *) malloc(num_outputs * sizeof(double));
      if (!layer -> weights[i]) {
        perror("Memory allocation error");
        return true;
      }
      for (int j = 0; j < num_outputs; j++) {
        layer -> weights[i][j] = ANN_RANDOM();
      }
    }

    layer -> prev -> next = layer;
  }

  return false;


}
  

/* Frees a given layer. */
void layer_free(layer_t *layer)
{
  free(layer->outputs);
  if (layer -> prev) {
    free(layer->biases);
    free(layer->deltas);
    for (int i = 0; i < layer -> num_inputs; i++) {
      free(layer -> weights[i]);
    }
    free(layer -> weights);
  }
  free(layer);
  
}

/* Computes the outputs of the current layer. */
void layer_compute_outputs(layer_t const *layer)
{
  if (!layer -> prev) {
    perror("Can not compute the outputs of the input layer");
    return;
  }

  for (int j = 0; j < layer->num_outputs; j++) {
    double sum = 0;
    for (int i = 0; i < layer->num_inputs; i++) {
      sum += (layer->weights[i][j] * layer->prev->outputs[i]);
    }

    layer->outputs[j] = sigmoid(layer->biases[j] + sum);
  }
}

/* Computes the delta errors for this layer. */
void layer_compute_deltas(layer_t const *layer)
{
  for (int i =0; i < layer->num_outputs; i++) {
    double sum = 0;
    for (int j=0; j < layer->next->num_outputs; j++) {
      sum += (layer->next->weights[i][j] * layer->next->deltas[j]);
    }
    layer->deltas[i] = sigmoidprime(layer->outputs[i] * sum);
  }
}

/* Updates weights and biases according to the delta errors given learning rate. */
void layer_update(layer_t const *layer, double l_rate)
{
  for (int j=0; j < layer->num_outputs; j++) {
    layer->biases[j] += (l_rate * layer->deltas[j]);
    for (int i=0; i < layer->num_inputs; i++) {
      layer->weights[i][j] += (l_rate * layer->prev->outputs[i] * layer->deltas[j]);
    }
  }


  /**** PART 1 - QUESTION 7 ****/
  /* objective: update layer->weights and layer->biases */

  /* 1 MARK */
}
