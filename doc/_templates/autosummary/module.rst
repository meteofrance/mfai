{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:

{% block modules %}
{% if modules %}
.. autosummary::
   :toctree:
   :recursive:

   {% for item in modules %}
   ~{{ item }}
   {%- endfor %}
{% endif %}
{% endblock %}