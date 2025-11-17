; Method definitions
(method_declaration
  name: (identifier) @method.name
  parameters: (formal_parameters) @method.params
  body: (block)? @method.body) @method.def

; Class definitions
(class_declaration
  name: (identifier) @class.name
  superclass: (superclass)? @class.extends
  interfaces: (super_interfaces)? @class.implements
  body: (class_body) @class.body) @class.def

; Interface definitions
(interface_declaration
  name: (identifier) @interface.name
  body: (interface_body) @interface.body) @interface.def

; Import statements
(import_declaration
  (scoped_identifier) @import.name) @import.stmt

; Method invocations
(method_invocation
  name: (identifier) @call.target
  arguments: (argument_list) @call.args) @call.expr

; Field access
(field_access
  object: (_) @field.object
  field: (identifier) @field.name) @field.access

; Throws clause
(throws
  (type_identifier) @error.type) @error.throws
