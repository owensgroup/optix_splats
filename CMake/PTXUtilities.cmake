function(add_ptx_targets project names)
  # Target to copy PTX files
  set(LIBRARY_NAMES ${names})
  list(TRANSFORM LIBRARY_NAMES PREPEND ${project}_)
  add_custom_target(
    ${project}_copy_ptx ALL
    COMMENT "Copy PTX Files for ${project}"
    DEPENDS ${LIBRARY_NAMES})

  # Target to create PTX directory
  add_custom_command(TARGET ${project}_copy_ptx PRE_BUILD
    COMMENT "Create directory for PTX files for ${project}"
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${project}>/ptx)

  # Create PTX objects
  foreach(cur_name ${names})
    add_library(${project}_${cur_name} OBJECT rtx/src/${cur_name}.cu)
    set_target_properties(${project}_${cur_name} PROPERTIES CUDA_PTX_COMPILATION ON)
    add_dependencies(${project} ${project}_${cur_name})

    # Add current PTX to copy target
    add_custom_command(TARGET ${project}_copy_ptx POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:${project}_${cur_name}> $<TARGET_FILE_DIR:${project}>/ptx
      DEPENDS ${project}_${cur_name})
  endforeach()
endfunction()