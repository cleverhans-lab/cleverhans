def compute_certificate(self):
    """Function to compute the certificate associated with feasible solution."""
    self.set_differentiable_objective()
    self.get_full_psd_matrix()
    # TODO: replace matrix_inverse with functin which uses matrix-vector product
    projected_certificate = (
        self.scalar_f +
        0.5*tf.matmul(tf.matmul(tf.transpose(self.vector_g),
                                tf.matrix_inverse(self.matrix_h)),
                      self.vector_g))
    return projected_certificate

  def project_dual(self):
    """Function that projects the input dual variables onto the feasible set.

    Returns:
      projected_dual: Feasible dual solution corresponding to current dual
      projected_certificate: Objective value of feasible dual
    """
    # TODO: consider whether we can use shallow copy of the lists without
    # using tf.identity
    projected_lambda_pos = [tf.identity(x) for x in self.lambda_pos]
    projected_lambda_neg = [tf.identity(x) for x in self.lambda_neg]
    projected_lambda_quad = [tf.identity(x) for x in self.lambda_quad]
    projected_lambda_lu = [tf.identity(x) for x in self.lambda_lu]
    projected_nu = tf.identity(self.nu)

    # TODO: get rid of the special case for one hidden layer
    # Different projection for 1 hidden layer
    if self.nn_params.num_hidden_layers == 1:
      # Creating equivalent PSD matrix for H by Schur complements
      diag_entries = 0.5*tf.divide(
          tf.square(self.lambda_quad[self.nn_params.num_hidden_layers]),
          (self.lambda_quad[self.nn_params.num_hidden_layers] +
           self.lambda_lu[self.nn_params.num_hidden_layers]))
      # If lambda_quad[i], lambda_lu[i] are 0, entry is NaN currently,
      # but we want to set that to 0
      diag_entries = tf.where(tf.is_nan(diag_entries),
                              tf.zeros_like(diag_entries), diag_entries)
      matrix = (
          tf.matmul(tf.matmul(tf.transpose(
              self.nn_params.weights[self.nn_params.num_hidden_layers-1]),
                              utils.diag(diag_entries)),
                    self.nn_params.weights[self.nn_params.num_hidden_layers-1]))
      new_matrix = utils.diag(
          2*self.lambda_lu[self.nn_params.num_hidden_layers - 1]) - matrix
      # Making symmetric
      new_matrix = 0.5*(new_matrix + tf.transpose(new_matrix))
      eig_vals = tf.self_adjoint_eigvals(new_matrix)
      min_eig = tf.reduce_min(eig_vals)
      # If min_eig is positive, already feasible, so don't add
      # Otherwise add to make PSD [1E-6 is for ensuring strictly PSD (useful
      # while inverting)
      projected_lambda_lu[0] = (projected_lambda_lu[0] +
                                0.5*tf.maximum(-min_eig, 0) + 1E-6)

    else:
      # Minimum eigen value of H
      # TODO: Write this in terms of matrix multiply
      # matrix H is a submatrix of M, thus we just need to extend existing code
      # for computing matrix-vector product (see get_psd_product function).
      # Then use the same trick to compute smallest eigenvalue.
      eig_vals = tf.self_adjoint_eigvals(self.matrix_h)
      min_eig = tf.reduce_min(eig_vals)

      for i in range(self.nn_params.num_hidden_layers+1):
        # Since lambda_lu appears only in diagonal terms, can subtract to
        # make PSD and feasible
        projected_lambda_lu[i] = (projected_lambda_lu[i] +
                                  0.5*tf.maximum(-min_eig, 0) + 1E-6)
        # Adjusting lambda_neg wherever possible so that lambda_neg + lambda_lu
        # remains close to unchanged
        # projected_lambda_neg[i] = tf.maximum(0.0, projected_lambda_neg[i] +
        #                                     (0.5*min_eig - 1E-6)*
        #                                     (self.lower[i] + self.upper[i]))

    projected_dual_var = {'lambda_pos': projected_lambda_pos,
                          'lambda_neg': projected_lambda_neg,
                          'lambda_lu': projected_lambda_lu,
                          'lambda_quad': projected_lambda_quad,
                          'nu': projected_nu}
    projected_dual_object = DualFormulation(projected_dual_var,
                                            self.nn_params,
                                            self.test_input,
                                            self.true_class,
                                            self.adv_class,
                                            self.input_minval,
                                            self.input_maxval,
                                            self.epsilon)
    projected_certificate = projected_dual_object.compute_certificate()
    return projected_certificate
