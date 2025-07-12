from flask import Flask,render_template,request
import model 
app = Flask('__name__')

valid_userid = ['00sab00','1234','zippy','zburt5','joshua','dorothy w','rebecca','walker557','samantha','raeanne','kimmie','cassie','moore222']
@app.route('/')
def view():
    return render_template('index.html')

@app.route('/recommend',methods=['POST'])
def recommend_top5():
    print(request.method)
    user_name = request.form['User Name']
    print('User name=',user_name)
    
    if  user_name in valid_userid and request.method == 'POST':
            top5_products = model.get_top5_user_recommendations(user_name)
            
            print(top5_products)
            top5_products = top5_products.reset_index()

            

           
           # return render_template('index.html',tables=[top5_products.to_html(classes='data',header=False,index=False)],text='Recommended products')
            #return render_template('index.html','top5_products['name'].tolist())
            #return render_template('index.html', names=top5_products['name'].values)

            return render_template('index.html',column_names=top5_products.name.values, row_data=list(top5_products.values.tolist()), zip=zip,text='Recommended products')
    elif not user_name in  valid_userid:
        return render_template('index.html',text='No Recommendation found for the user')
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.debug=False

    app.run()