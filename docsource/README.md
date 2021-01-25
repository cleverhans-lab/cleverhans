# Generate documentation

To generate the documentation do:
`make github`

The documentation files will be copied to the `cleverhans/docs` directory.

### Preparation

Please do:
`pip install sphinx`

Add a `.nojekyll` file in the `cleverhans/docs` directory. When GitHub sees
a `.nojekyll` file, it serves the root `index.html` file. The `.nojekyll` file
indicates that we are not using Jekyll as our static site generator in this
repository.

### Enable GitHub Pages for the GitHub repository

1. Go to the repository on the GitHub website and make sure you are logged in.
2. Add a /docs directory to the master branch. Otherwise you do not get the
   master branch /docs folder for the Source option in the drop-down list.
3. Click the Settings tab. You first go to the Options section.
4. Scroll down to the GitHub Pages section and choose the drop-down list under
   Source. Note: Your choices will differ based on whether you’re in a User repo
   or an Org repository.
5. To keep source and output HTML separate, choose master branch /docs folder
   for Source.

### Build Sphinx locally and publish on GitHub Pages

We keep the source docsource and output docs separate, but still are able to
publish on GitHub Pages and preview builds locally.

We have the following option in the Makefile:

```
  github:
      @make html
      @cp -a _build/html/. ../docs
```

Thus, we can run `make github` from the `docsource` directory to generate a
local preview and move the docs where GitHub wants to serve them from.

### Hacks

If you cannot build the docs for attacks, uncomment
`import tensorflow_addons as tfa` in `cleverhans/attacks/spsa.py`.

Otherwise:

```angular2html
WARNING: autodoc: failed to import module 'attacks' from module 'cleverhans'; the following exception was raised:
cannot import name 'keras_tensor'
```

It is convenient to create a virtual environment to install all the specific
libraries (e.g. virutalen cleverhans).
