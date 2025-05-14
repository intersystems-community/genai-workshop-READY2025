import iris
import requests
from create_desc_vectors import load_data, vectorize_data, load_model, add_embedding_config

# switch namespace to the %SYS namespace
iris.system.Process.SetNamespace("%SYS")

# set credentials to not expire
iris.cls('Security.Users').UnExpireUserPasswords("*")

# switch namespace to IRISAPP built by merge.cpf
iris.system.Process.SetNamespace("IRISAPP")

iris.cls('%SYSTEM.OBJ').Import("/home/irisowner/dev/src/GenAI/encounters.xml", "ck")
 

#iris.cls('%ZPM.PackageManager').Shell("load /home/irisowner/dev -v")
#assert iris.cls('%IPM.PackageManager').Load("/home/irisowner/dev")
#s version="latest" s r=##class(%Net.HttpRequest).%New(),r.Server="pm.community.intersystems.com",r.SSLConfiguration="ISC.FeatureTracker.SSL.Config" d r.Get("/packages/zpm/"_version_"/installer"),$system.OBJ.LoadStream(r.HttpResponse.Data,"c")
#iris.cls('IPM.Installer').GetPackageList()
#assert iris.cls('%IPM.PackageManager').Shell("load /home/irisowner/dev -v")

#assert iris.ipm('load /home/irisowner/dev -v')


table_name = "GenAI.encounters"
data = load_data()
# load_model()
add_embedding_config(delete=False)
vectorize_data(data, table_name)