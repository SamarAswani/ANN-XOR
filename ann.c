#include "ann.h"

/* Creates and returns a new ann. */
ann_t *ann_create(int num_layers, int *layer_outputs)
{
  ann_t *ann = (ann_t *) calloc(1, sizeof(ann_t));
  if (!ann) {
    perror("Memory allocation error");
    return NULL;
  }
  ann->input_layer = layer_create();
  if (!ann->input_layer) {
    perror("Memory allocation error");
    return NULL;
  }
  layer_init(ann->input_layer, layer_outputs[0], NULL);
  layer_t *prev = ann->input_layer;

  for (int i = 1; i < num_layers-1; i++) {
     layer_t *tempLayer = layer_create();
     if (!tempLayer) {
        perror("Memory allocation error");
        return NULL;
    }
     layer_init(tempLayer, layer_outputs[i], prev);
     prev->next = tempLayer;
     prev = tempLayer;
  }

  ann->output_layer = layer_create();
  if (!ann->output_layer) {
    perror("Memory allocation error");
    return NULL;
  }
  layer_init(ann->output_layer, layer_outputs[num_layers-1], prev);
  prev->next = ann->output_layer;

}

/* Frees the space allocated to ann. */
void ann_free(ann_t *ann)
{
  layer_t *temp = ann->input_layer->next;
  layer_free(ann->input_layer);
  while(temp != ann->output_layer) {
    temp = temp->next;
    layer_free(temp->prev);
  }
  layer_free(ann->output_layer);
}

/* Forward run of given ann with inputs. */
void ann_predict(ann_t const *ann, double const *inputs)
{
  for (int i = 0; i < ann -> input_layer -> num_outputs; i++) {
    ann -> input_layer -> outputs[i] = inputs[i];
  }
  layer_t *temp = ann->input_layer->next;
  while(temp != ann->output_layer) {
    layer_compute_outputs(temp);
    temp = temp->next;
  }
  layer_compute_outputs(ann->output_layer);
}

/* Trains the ann with single backprop update. */
void ann_train(ann_t const *ann, double const *inputs, double const *targets, double l_rate)
{
  /* Sanity checks. */
  assert(ann != NULL);
  assert(inputs != NULL);
  assert(targets != NULL);
  assert(l_rate > 0);

  /* Run forward pass. */
  ann_predict(ann, inputs);

  for (int i = 0; i < ann->output_layer->num_outputs; i++) {
    ann->output_layer->deltas[i] = sigmoid(ann->output_layer->outputs[i]) * (targets[i] - ann->output_layer->outputs[i]);
  }
  layer_t *temp = ann->output_layer->prev;
  while(temp != ann->input_layer) {
    layer_compute_deltas(temp);
    temp = temp->prev;
  }

  while (temp != ann -> output_layer) {
    temp = temp -> next;
    layer_update(temp, l_rate);
  }

  /* 3 MARKS */
}
