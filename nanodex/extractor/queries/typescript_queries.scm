; Function declarations
(function_declaration
  name: (identifier) @func.name
  parameters: (formal_parameters) @func.params
  body: (statement_block) @func.body) @func.def

; Arrow functions
(arrow_function
  parameters: [
    (identifier) @func.param
    (formal_parameters) @func.params
  ]
  body: [
    (statement_block) @func.body
    (_) @func.expr
  ]) @func.arrow

; Class declarations
(class_declaration
  name: (type_identifier) @class.name
  (class_heritage
    (extends_clause
      value: (_) @class.extends)?
    (implements_clause
      (_) @class.implements)?)? @class.heritage
  body: (class_body) @class.body) @class.def

; Interface declarations
(interface_declaration
  name: (type_identifier) @interface.name
  (extends_clause
    (_) @interface.extends)?
  body: (object_type) @interface.body) @interface.def

; Import statements
(import_statement
  source: (string) @import.source) @import.stmt

; Export statements
(export_statement) @export.stmt

; Call expressions
(call_expression
  function: [
    (identifier) @call.target
    (member_expression) @call.target
  ]
  arguments: (arguments) @call.args) @call.expr

; Method definitions
(method_definition
  name: [
    (property_identifier) @method.name
    (computed_property_name) @method.name
  ]
  parameters: (formal_parameters) @method.params
  body: (statement_block) @method.body) @method.def
