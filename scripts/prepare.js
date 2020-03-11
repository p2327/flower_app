az.load_font('Raleway')
az.style_body({
    "font-family": "Raleway",
    "min-width": "1300px",
    "max-width": "1300px",
    "align": "center"
})
az.add_top_button({
    "this_class": "scroll_to_top",
    "text": "TOP",
    "side": "right",
    "spin": true
})
az.add_sections({
    "this_class": "my_sections",
    "sections": 4
})
az.all_style_sections('my_sections', {
    "height": "500px",
    "background": "CadetBlue"
    //"width": "800px"
})
az.style_sections('my_sections', 1, {
    "height": "60px"
})
az.add_layout('my_sections', 1, {
    "this_class": "banner_layout",
    "row_class": "banner_layout_rows",
    "cell_class": "banner_layout_cells",
    "number_of_rows": 1,
    "number_of_columns": 3
})
az.style_layout('banner_layout', 1, {
    "height": "auto",
    "width": "100%",
    "column_widths": ['10%', '20%', '70%'],
    // remove the border
    "border": 0
})
az.add_image('banner_layout_cells', 1, {
    "this_class": "logo",
    "image_path": "http://realerthinks.com/wordpress/wp-content/uploads/2017/03/TF_iris.png"
})
az.style_image('logo', 1, {
    "width": "50px",
    "margin-top": "4px",
    "margin-left": "8px"
})
az.add_text('banner_layout_cells', 2, {
    "this_class": "app_title",
    "text": "Tensor Flower",
})
az.style_text('app_title', 1, {
    "font-size": "20px",
    "margin-left": "4px",
    "text-shadow" : "1px 1px 1px black",
    "color" : "orange",
})
az.style_layout('banner_layout_cells', 2, {
    "halign": "left"
})
az.style_layout('banner_layout_cells', 3, {
    "halign": "right"
})

az.hold_value.nav_button_text = ['TRAIN', 'CHALLENGE', 'LEADERBOARD']

az.call_multiple({
    "iterations": 3,
    "function": `
            az.add_button('banner_layout_cells', 3, {
                "this_class": "nav_buttons",
                "text": az.hold_value.nav_button_text[index]
            })
            az.all_style_button('nav_buttons', {
                "background" : "black",
                "color" : "orange",
                "border" : "1px solid orange",
                "margin-top" : "0px",
                "margin-left" : "4px",
                "margin-right" : "4px",
            })
            scroll_index = index + 2
            az.add_event('nav_buttons', index+1, {
                "type" : "click",
                "function" : "az.scroll_to('my_sections', " + scroll_index + ")"
            })
            `
})


