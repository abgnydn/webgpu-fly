/**
 * flybody never ships the actuator filter dynamics in the MJCF; it applies
 * them programmatically after parsing, in
 * flybody/fruitfly/fruitfly.py:_build (joint_filter=0.01, adhesion_filter=
 * 0.007, dyntype='filter'):
 *
 *     if joint_filter > 0:
 *         for actuator in root.find_all('actuator'):
 *             if actuator.tag != 'adhesion':
 *                 actuator.dyntype = dyntype
 *                 actuator.dynprm = (joint_filter, )
 *     if adhesion_filter > 0:
 *         for actuator in root.find_all('actuator'):
 *             if actuator.tag == 'adhesion':
 *                 actuator.dclass.parent.general.dyntype = dyntype
 *                 actuator.dclass.parent.general.dynprm = (adhesion_filter, )
 *
 * We read the shipped XML, so we have to reproduce that here or the plant is
 * ~5.5x stiffer per control tick than the one the policy was trained on and
 * the 59-dim actuator_activation observable is all zeros.
 *
 * The two rewrites mirror the two loops above: `<general name=` matches only
 * the 70 joint actuators (defaults-block `<general>` elements are unnamed),
 * and the adhesion class default is the exact element fruitfly.py reaches
 * through `dclass.parent.general`.
 */
export function patchActuatorFilters(xml: string): string {
  const JOINT = /<general name=/g;
  const nJoint = (xml.match(JOINT) ?? []).length;
  const ADHESION = '<general dyntype="none" dynprm="1"/>';
  if (nJoint !== 70 || !xml.includes(ADHESION)) {
    throw new Error(`fruitfly.xml actuator filter patch found ${nJoint} joint actuators, adhesion default ${xml.includes(ADHESION)}`);
  }
  return xml
    .replace(JOINT, '<general dyntype="filter" dynprm="0.01" name=')
    .replace(ADHESION, '<general dyntype="filter" dynprm="0.007"/>');
}
