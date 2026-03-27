{{ objname | escape }}
{{ "=" * (objname | escape | length) }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

.. image:: model_diagrams/{{ module }}.svg