; Function definitions
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @func.name
    parameters: (parameter_list) @func.params)
  body: (compound_statement) @func.body) @func.def

; Class definitions
(class_specifier
  name: (type_identifier) @class.name
  (base_class_clause)? @class.bases
  body: (field_declaration_list) @class.body) @class.def

; Struct definitions
(struct_specifier
  name: (type_identifier) @struct.name
  body: (field_declaration_list) @struct.body) @struct.def

; Include directives
(preproc_include
  path: [
    (string_literal) @include.path
    (system_lib_string) @include.path
  ]) @include.stmt

; Namespace definitions
(namespace_definition
  name: (identifier)? @namespace.name
  body: (declaration_list) @namespace.body) @namespace.def

; Call expressions
(call_expression
  function: [
    (identifier) @call.target
    (field_expression) @call.target
  ]
  arguments: (argument_list) @call.args) @call.expr

; Throw expressions
(throw_statement
  (_) @error.expr) @error.throw
