{{ objname | escape }}
{{ "=" * (objname | escape | length) }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

.. raw :: html

   <img src="{{ pathto(objname + '.svg', 1) }}" alt="" />